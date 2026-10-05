# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Report the structure of an ICON netCDF dataset as a small JSON document.

Intended for datasets that are too large to move: run this where the data
lives, and the few kilobytes of JSON it prints are enough to write a reader
against. It deliberately reads only a small contiguous slab of each variable,
so it stays cheap even on a multi terabyte archive.

What it reports and why each part matters for the reader:

* dimensions and variable shapes, to tell ``(time, plev, ncells)`` layouts from
  ``(time, ncells)`` ones and to find the cell dimension;
* the HDF5 filter pipeline per variable, which ``ncdump`` does not show in full:
  plain zlib is transparent to h5py, while zstd, blosc or szip need
  ``hdf5plugin`` to be importable before the file will read at all;
* chunk shapes, which decide whether levels can be subset cheaply and whether
  streaming reads are viable at training time;
* ``scale_factor``/``add_offset``/``_FillValue``, which h5py does *not* apply,
  so a packed variable reads back as raw integers;
* the ``units`` of the time coordinate plus its first raw values, to pin down
  which of the two time encodings the files use;
* value ranges from a small slab, which catch unit surprises such as
  geopotential in m2/s2 versus geopotential height in m.

Usage::

    python data_process/probe_icon_dataset.py --file <data.nc> [--grid_file <grid.nc>] \\
        [--directory <one_day_of_files>] [--output report.json]

Only h5py and numpy are required; no MPI, no netCDF library.
"""

import os
import glob
import json
import argparse as ap

import numpy as np
import h5py


# a slab this size is enough to characterize a variable while touching only the
# first few chunks of it
SLAB_ELEMENTS = 100_000

# HDF5 filter ids that h5py handles without hdf5plugin
_BUILTIN_FILTERS = {1: "deflate/zlib", 2: "shuffle", 3: "fletcher32", 4: "szip", 5: "nbit", 6: "scaleoffset"}


def _jsonable(value):
    """Convert numpy scalars, arrays and byte strings into JSON friendly values."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size > 16:
            return [_jsonable(v) for v in value.reshape(-1)[:16]] + ["..."]
        return [_jsonable(v) for v in value.reshape(-1)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attributes(obj):
    return {key: _jsonable(value) for key, value in obj.attrs.items()}


def _filters(dset):
    """Return the HDF5 filter pipeline of a dataset.

    Read from the dataset creation property list rather than from the
    convenience attributes, because those only name the filters h5py knows: an
    unregistered codec shows up here as its numeric id, which is exactly the
    case we need to detect.
    """
    plist = dset.id.get_create_plist()
    filters = []
    for index in range(plist.get_nfilters()):
        code, _flags, cd_values, name = plist.get_filter(index)
        filters.append(
            {
                "id": int(code),
                "name": _jsonable(name) or _BUILTIN_FILTERS.get(int(code), "unknown"),
                "needs_hdf5plugin": int(code) not in _BUILTIN_FILTERS,
                "options": [int(v) for v in cd_values],
            }
        )
    return filters


def _slab(dset):
    """Read a small leading slab of a dataset, touching as few chunks as possible."""
    if dset.size == 0:
        return None

    selection = []
    remaining = SLAB_ELEMENTS
    for axis, length in enumerate(dset.shape):
        if axis < dset.ndim - 1:
            # leading axes (time, level): take a single index
            selection.append(slice(0, 1))
        else:
            selection.append(slice(0, int(min(length, remaining))))
    return dset[tuple(selection)]


def _statistics(values, attrs):
    """Summarize a slab, both as stored and as it would look decoded."""
    if values is None:
        return None

    flat = np.asarray(values).reshape(-1)
    if not np.issubdtype(flat.dtype, np.number):
        return {"dtype": str(flat.dtype), "note": "non numeric"}

    as_float = flat.astype(np.float64)
    report = {
        "sampled_elements": int(flat.size),
        "raw_min": float(np.nanmin(as_float)),
        "raw_max": float(np.nanmax(as_float)),
        "raw_mean": float(np.nanmean(as_float)),
        "first_values": [float(v) for v in as_float[:5]],
    }

    fill = attrs.get("_FillValue", attrs.get("missing_value"))
    if fill is not None:
        report["fill_value_fraction"] = float(np.mean(flat == np.asarray(fill).reshape(-1)[0]))

    scale = attrs.get("scale_factor")
    offset = attrs.get("add_offset")
    if scale is not None or offset is not None:
        decoded = as_float * float(scale if scale is not None else 1.0) + float(offset if offset is not None else 0.0)
        report["packed"] = True
        report["decoded_min"] = float(np.nanmin(decoded))
        report["decoded_max"] = float(np.nanmax(decoded))
    else:
        report["packed"] = False

    return report


def probe_file(path: str, with_values: bool = True) -> dict:
    """Describe a single netCDF/HDF5 file."""
    report = {
        "path": os.path.basename(path),
        "size_bytes": os.path.getsize(path),
        "global_attributes": {},
        "variables": {},
    }

    with h5py.File(path, "r") as handle:
        report["global_attributes"] = _attributes(handle)

        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            attrs = _attributes(obj)
            entry = {
                "shape": [int(s) for s in obj.shape],
                "dtype": str(obj.dtype),
                "chunks": [int(c) for c in obj.chunks] if obj.chunks else None,
                "filters": _filters(obj),
                "attributes": attrs,
            }
            if with_values:
                try:
                    entry["values"] = _statistics(_slab(obj), obj.attrs)
                except Exception as err:  # a codec we cannot decode is itself the finding
                    entry["values"] = {"error": f"{type(err).__name__}: {err}"}
            report["variables"][name] = entry

        handle.visititems(visit)

        # the time coordinate decides which of the two encodings the files use, so
        # its raw values are reported in full rather than sampled
        for name in ("time", "Time", "valid_time"):
            if name in handle and handle[name].size <= 1024:
                report["time_values_raw"] = [_jsonable(v) for v in handle[name][:]]
                break

    return report


def probe_directory(path: str, limit: int = 40) -> dict:
    """Describe the file layout of a directory, to show naming and cadence."""
    # filter to files before truncating, so that a directory entry cannot push a
    # file out of the listing, and so the count means what it says
    files = [entry for entry in sorted(glob.glob(os.path.join(path, "*"))) if os.path.isfile(entry)]
    listing = [{"name": os.path.basename(entry), "size_bytes": os.path.getsize(entry)} for entry in files[:limit]]
    return {
        "path": path,
        "file_count": len(files),
        "listed": listing,
        "truncated": len(files) > limit,
    }


def probe_grid_layout(path: str, blocks: int = 24, samples: int = 20000) -> dict:
    """Characterize how the flat cell axis maps onto the sphere.

    ICON stores the horizontal grid as a single ``ncells`` axis with no 2-D
    structure, so how cell *index* relates to cell *position* is not visible
    from the header, yet it decides how the data can be read. If consecutive
    indices are spatially local, a rank that owns a lat/lon block can read a few
    contiguous ranges; if the ordering is scattered, it has to touch the whole
    array or gather millions of separate elements.

    The cell centres are sampled on a stride and the array is split into
    ``blocks`` equal index ranges, each reported with the bounding box of the
    cells sampled inside it. A block covering a small box means index order
    follows the grid hierarchy; boxes spanning the globe mean it does not.

    ``span_fraction`` is the bounding box area as a fraction of the sphere, so
    ``1/blocks`` is the ideal (perfectly local) value and 1.0 means the block is
    spread over everything.
    """
    with h5py.File(path, "r") as handle:
        if "clon" not in handle or "clat" not in handle:
            return {"error": "no clon/clat in this file; use one that carries the coordinates"}

        ncells = int(handle["clon"].shape[0])
        stride = max(1, ncells // max(1, samples))

        lon = np.rad2deg(np.asarray(handle["clon"][::stride], dtype=np.float64))
        lat = np.rad2deg(np.asarray(handle["clat"][::stride], dtype=np.float64))

    lon = np.mod(lon, 360.0)
    indices = np.arange(0, ncells, stride)[: lon.size]

    def _box(lon_sel, lat_sel):
        if lon_sel.size == 0:
            return None
        lon_min, lon_max = float(lon_sel.min()), float(lon_sel.max())
        lat_min, lat_max = float(lat_sel.min()), float(lat_sel.max())
        # area of the lat/lon box as a fraction of the sphere; the sin() terms
        # weight by the cosine of latitude, so polar boxes are not overstated
        span = ((lon_max - lon_min) / 360.0) * abs(np.sin(np.deg2rad(lat_max)) - np.sin(np.deg2rad(lat_min))) / 2.0
        return {
            "lon_min": round(lon_min, 3),
            "lon_max": round(lon_max, 3),
            "lat_min": round(lat_min, 3),
            "lat_max": round(lat_max, 3),
            "span_fraction": round(float(span), 5),
        }

    edges = np.linspace(0, ncells, blocks + 1).astype(np.int64)
    block_reports = []
    for start, end in zip(edges[:-1], edges[1:]):
        selection = (indices >= start) & (indices < end)
        block_reports.append(
            {
                "index_start": int(start),
                "index_end": int(end),
                "sampled": int(selection.sum()),
                "bounds": _box(lon[selection], lat[selection]),
            }
        )

    return {
        "ncells": ncells,
        "sample_stride": int(stride),
        "sampled_cells": int(lon.size),
        "global_bounds": _box(lon, lat),
        "ideal_span_fraction_per_block": round(1.0 / blocks, 5),
        "blocks": block_reports,
    }


def main(args):
    report = {}

    if args.file:
        report["data_file"] = probe_file(args.file, with_values=not args.no_values)

    if args.grid_layout:
        report["grid_layout"] = probe_grid_layout(args.grid_layout, blocks=args.layout_blocks)

    if args.grid_file:
        report["grid_file"] = probe_file(args.grid_file, with_values=not args.no_values)

    if args.directory:
        report["directory"] = probe_directory(args.directory)

    text = json.dumps(report, indent=2, sort_keys=True, default=str)

    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text)
        print(f"wrote {args.output} ({len(text)} bytes)")
    else:
        print(text)


if __name__ == "__main__":
    parser = ap.ArgumentParser(description=__doc__, formatter_class=ap.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=str, default=None, help="An ICON data file to describe.")
    parser.add_argument("--grid_file", type=str, default=None, help="The matching ICON grid file.")
    parser.add_argument("--directory", type=str, default=None, help="Directory to list, to show naming and cadence.")
    parser.add_argument("--output", type=str, default=None, help="Write the report here instead of to stdout.")
    parser.add_argument("--no_values", action="store_true", help="Skip reading data, report structure only.")
    parser.add_argument(
        "--grid_layout",
        type=str,
        default=None,
        help="File carrying clon/clat; reports how the flat cell axis maps onto the sphere.",
    )
    parser.add_argument(
        "--layout_blocks", type=int, default=24, help="Number of index ranges to characterize for --grid_layout."
    )
    args = parser.parse_args()

    if not (args.file or args.grid_file or args.directory or args.grid_layout):
        parser.error("nothing to do: pass at least one of --file, --grid_file, --directory, --grid_layout")

    main(args)
