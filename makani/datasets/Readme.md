# Datasets

This package contains the scripts that build makani datasets: converting ERA5, annotating and concatenating the files, and computing their statistics. Each runs as a module, `python -m makani.datasets.<script>`, and their dependencies come with `pip install makani[datasets]`.

## Running the scripts

MPI is optional:
- **Serially**, with plain `python -m makani.datasets.<script>`: the script runs on a single rank and works with any h5py, including the pip wheels.
- **In parallel**, under `mpirun -n <ranks> python -m makani.datasets.<script>`, after `pip install makani[datasets-mpi]`, which adds mpi4py. The scripts that write a shared file from several ranks, `convert_era5`, `generate_wb2_climatology` and `merge_channels`, then also need h5py built against parallel HDF5, and stop with a message saying so if it is not.

The statistics scripts, `compute_stats`, `compute_spectra` and `compute_hydrostatic_balance_climatology`, reduce with torch.distributed: over NCCL on GPUs, one per rank, and over gloo on CPUs, so they need no GPU. They listen on `MASTER_PORT`, 29500 unless set.

The `mpirun -n 8` in the examples below can be dropped for a serial run.

## Overview of processing

ERA5 data is provided in .h5 file format. Often data for several years and several variables is split up into files for each respective year, then containing the data for all variables, all timepoints during that year and all spatial locations. Each h5 file represents a dataset that contains additional metadata relevant for the processing of the data.

### Directory structure
This folder is organized as follows:

```
makani
├── ...
├── datasets                                        # building makani datasets
│   ├── annotate_dataset.py                         # annotation of the dataset
│   ├── compute_histograms.py                       # compute histograms for each variable over a dataset
│   ├── compute_hydrostatic_balance_climatology.py  # climatology for the hydrostatic balance constraint
│   ├── compute_spectra.py                          # compute power spectra for each variable over a dataset
│   ├── compute_stats.py                            # compute the normalization statistics of a dataset
│   ├── concatenate_dataset.py                      # concatenation of data files across several years
│   ├── convert_era5.py                             # convert ERA5 (WB2/ARCO or NSF NCAR) into makani's yearly files
│   ├── convert_era5_invariants.py                  # convert time invariant ERA5 fields (WB2/ARCO or NSF NCAR) into makani format
│   ├── convert_makani_output_to_wb2.py             # convert makani inference output to WeatherBench2 format
│   ├── date_range.py                               # date range options shared by the yearly converters
│   ├── examples/metadata.json                      # an example of the metadata file of a dataset
│   ├── generate_wb2_climatology.py                 # generate mask and dataset for climatology data
│   ├── helpers.py                                  # distributed Welford reductions, progress bar, wind channels
│   ├── merge_channels.py                           # copy channels from one makani dataset into another
│   ├── parallel.py                                 # optional MPI: the communicator, serially without mpi4py
│   ├── postprocess_stats.py                        # clean up computed statistics: water minima, std clamping
│   ├── rechunk_dataset.py                          # rewrite a dataset with new chunking and compression
│   ├── sources                                     # ERA5 sources of the two convert_era5* scripts (wb2, ncar)
│   └── Readme.md                                   # this file
...
```

### Annotate dataset

For scoring, the .h5 files are expected to be annotated with the correct metadata and dates. `annotate_dataset.py` modifies the files such that all relevant metadata is contained in the dataset, and that these metadata is universally equal across different dataset. Here, the original data is read, and for each file, metadata timestamps, latitude, longitude and channel are copied from the original file. Labels are provided in a uniform way. Timestamps are edited by converting the start sample into UTC time zone and deriving all following time samples from the UTC converted one. This ensure a unform time information across datasets.

### Concatenate dataset

`concatenate_dataset.py` creates a virtual dataset by combining several .h5 files into a single (virtual) dataset. This virtual dataset represents data from several years.

### Compute statistics and histograms

`compute_stats.py ` several statistics calculated for either a folder containing several h5 files, or a combined h5f dataset (virtual dataset, concatenated across several years). Calculated stats are: global_means, global_stds, mins, maxs, time-means, time_diff_means_dt, time_diff_stds_dt. In a similar fashion, `compute_histograms.py` computes histograms from the dataset

### Weatherbench

Makani contains several files to enable scoring consistent with Weatherbench2. `generate_wb2_climatology.py` computes climatology data provided by WeatherBench2 (ERA5 data, averaged data from 1990 - 2019) and converts them to a h5 dataset. Additionally, generates a climatology masks used by WB. Other helper functions are contained in `makani/data/wb2_helpers.py`. `convert_era5.py wb2` can be used to convert Weatherbench2 data such as the ARCO-ERA5 dataset to a makani-compatible format. `convert_makani_output_to_wb2.py` converts makani inference output to Weatherbench2. `merge_channels.py` copies channels from one makani dataset into another, for example channels such as the 10 m winds that are in Copernicus ERA5 but not in ARCO-ERA5.

## Data Processing Examples

### Date ranges

`convert_era5.py` writes one Makani file per year, whichever source it
reads from (`wb2` or `ncar`, given as subcommand). It selects the samples with
`--start_date` and `--end_date`, both inclusive and in UTC. They take ISO 8601
dates such as `2018-01-01` or `2018-01-01T06`, and a bare end date includes the whole day.

- One file `<year>.h5` is written for every year the range touches. The first and last of
  them are partial if the range starts or ends mid-year, which is how a year that is still
  being published upstream gets converted.
- Samples always sit on the `dhours` grid anchored at 00Z on January 1st, so a start date
  off that grid is rounded up to the next sample.
- An existing yearly file is skipped, partial or not. With `--force_overwrite` it is
  replaced by exactly the samples of the new range; nothing from the old file is kept.
  To extend a partial year, rerun with a range covering the whole period you want in the
  file, so that every file stays free of gaps.

### Creating a Makani dataset from Weatherbench (ARCO-ERA5):

Below is a minimal end-to-end example for converting ARCO-ERA5 (Weatherbench2) Zarr
data into Makani-compatible yearly `.h5` files.

1. Create a metadata json describing cadence and coordinates (example):
   ```json
   {
     "dhours": 6,
     "coords": {
       "lat": [-90.0, ..., 90.0],
       "lon": [0.0, ..., 360.0],
       "channel": ["t2m", "u10", "v10", "z500", "t500", "q500"]
     }
   }
   ```
   A full example for a metadata file for a dataset which was used to train FourCastNet3 can be found under `examples/metadata.json`.
2. Run the converter, here on 8 ranks; without `mpirun -n 8` it runs serially, see [Running the scripts](#running-the-scripts):
   ```
   mpirun -n 8 python -m makani.datasets.convert_era5 wb2 \
     --input_file "gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2" \
     --output_dir "/path/to/output/makani_era5" \
     --metadata_file "/path/to/metadata.json" \
     --start_date 2018-01-01 \
     --end_date 2019-12-31 \
     --batch_size 8 \
     --skip_missing_channels \
     --impute_missing_timestamps
   ```
   Flags you may want to use:
   - `--force_overwrite` to replace existing yearly files. See [Date ranges](#date-ranges).
   - `--coord_mode` controls how input lat/lon are aligned to the metadata grid:
     - `match` (default): use xarray `.sel()` to reorder input coordinates to match the metadata file.
     - `force-flip-lat`: flip the latitude axis of the input without coordinate matching.
     - `force`: read data as-is, no reordering — assumes input already matches metadata.
   - `--skip_missing_channels` writes channels absent from the store as missing (NaN, `valid_data` cleared).
     Omit it and `--impute_missing_timestamps` to fail fast on gaps.

The command writes one Makani-format `.h5` file per requested year in `output_dir`
with shared dimension scales (`timestamp`, `channel`, `lat`, `lon`) and a
`valid_data` mask that tracks imputed or missing values. Imputed values are NaN, and the
`fields` dataset declares NaN as its HDF5 fill value (`h5dump -p` shows it), so missing
data is self describing. The fill value is declaration only: HDF5 never prefills the
dataset, since under parallel HDF5 that would mean writing every yearly file twice.

### Creating a Makani dataset from NSF NCAR ERA5 (RDA d633000):

`convert_era5.py ncar` reads the public, anonymously accessible bucket
`s3://nsf-ncar-era5`, which hosts ERA5 from 1940 to the present as netCDF4 on its native
0.25 degree grid. That grid is already the grid makani expects (latitude 90 to -90,
longitude 0 to 359.75), so no regridding or latitude flipping takes place and a mismatch
against the metadata grid is reported as an error rather than silently corrected.

The `ncar` source takes the same metadata json as the `wb2` source and the converter writes
the same yearly files for both, so the two are interchangeable as dataset sources:
```
mpirun -n 8 python -m makani.datasets.convert_era5 ncar \
  --output_dir "/path/to/output/makani_era5" \
  --metadata_file "/path/to/metadata.json" \
  --start_date 2018-01-01 \
  --end_date 2019-12-31
```
See [Date ranges](#date-ranges) for how the range maps to files. NCAR publishes with a
lag, and the analysis streams on the bucket typically run a month or so ahead of the
forecast streams that `tp` is reconstructed from, so when converting the current year,
either end the range where the slower of the two stops, or end it with the analysis and
pass `--impute_missing_timestamps`, which leaves `tp` NaN and invalid for the gap.
Flags you may want to use:
- `--cache_dir` keeps the raw NCAR files on local disk, so an interrupted run and any
  later re-conversion do not refetch them. Note that the raw files are considerably
  larger than the converted output; without this flag nothing is staged locally.
- `--accumulation_hours` sets the window used for accumulated channels such as `tp`.
  It defaults to `dhours`, so a 6-hourly dataset gets 6-hourly accumulations.
- `--skip_missing_channels` writes channels that have no NCAR counterpart as missing (NaN, `valid_data`
  cleared) instead of failing.
- `--impute_missing_timestamps` writes NaN and clears `valid_data` for data that is not on
  the bucket instead of failing. Imputation works per channel group and day: if anything a
  group needs for a day is missing, all of that group's channels are imputed for the day.
- `--force_overwrite` to replace existing yearly files. See [Date ranges](#date-ranges).

Notes on the source data:
- Data is streamed with byte-range reads. The pressure level files are chunked as one
  chunk per timestep holding all 37 levels, so subsetting in time is nearly free while
  subsetting in level is not: a 6-hourly dataset transfers roughly a sixth of the bytes
  an hourly one would, but requesting fewer pressure levels does not reduce transfer.
- Work is distributed over whole days rather than individual timestamps, so that two
  ranks never fetch and decompress the same pressure level chunk.
- d633000 does not ship total precipitation, so `tp` is reconstructed as `lsp + cp`
  (large-scale plus convective precipitation, both in metres). Unlike the ECMWF forecast
  archive, these fields arrive already de-accumulated, one value per forecast hour, and
  the converter sums the hours spanned by the window. Windows longer than the 12 hour
  forecast run are stitched together from consecutive runs.
- Fields that are undefined over part of the globe, most notably `sst`, carry a netCDF
  `_FillValue` that the converter translates to NaN.

### Concatenate individual years into a single dataset

The script `concatenate_dataset.py` builds a virtual HDF5 dataset (VDS) that references the existing yearly Makani `.h5` files; no data is copied. The VDS stores the exact paths to the source files, so those paths must remain readable later (e.g.,
training or inference). If you run inside Docker, mount the yearly files at the
same absolute locations that were used when the VDS was created.

Basic usage (single directory of yearly files):
```
python -m makani.datasets.concatenate_dataset \
  --dataset_metadata /data/makani_era5/metadata.json \
  --input_dirs /data/makani_era5 \
  --output_file /data/makani_era5/makani_era5_vds.h5 \
  --dhours_rel 1          # 1 keeps every timestep; a value >1 employs temporal subsampling
```

Path consistency with Docker:
```
# Create VDS on the host (absolute paths are embedded in the VDS):
python -m makani.datasets.concatenate_dataset \
  --dataset_metadata /mnt/era5/metadata.json \
  --input_dirs /mnt/era5/train \
  --output_file /mnt/era5/makani_era5_train_vds.h5v # we give the file a differen suffix in order to distinguish it from "proper" h5 files

# Later training/inference must see the same paths: make sure train_data_path and valid_data_path are set to the concatenated file, i.e. change in the config.yaml:
train_data_path: /mnt/era5/makani_era5_train_vds.h5v
valid_data_path: /mnt/era5/makani_era5_valid_vds.h5v
```
And ensure that `/mnt/era5/train` and e.g. `/mnt/era5/valid` as well as `/mnt/era5` are visible in the docker container during training or validation.

You can combine multiple input directories (e.g., splitting channels across
files), pass one `--dataset_metadata` per `--input_dirs` in the same order. All
inputs must share `dhours`, latitude, longitude, and contain the same years. The concatenated dataset will then contain a virtual dataset with all channels and years. This enables users to augment datasets with more variables or years without having to download existing files again or moving a lot of data.

From performance perspective, there is no significant difference in IO time between a conventional HDF5 dataset and a virtual HDF5 dataset if all the data resides on the same storage.

### Compute Statistics on Dataset for Data Normalization

The script `compute_stats.py` computes per-channel statistics for a Makani dataset (yearly files or a virtual HDF5 created via `concatenate_dataset.py`). Outputs are NumPy files:
- `global_means.npy`, `global_stds.npy` (per-channel scalars)
- `mins.npy`, `maxs.npy` (per-channel scalars)
- `time_means.npy` (per-channel spatial climatology)
- `time_diff_means_dt<dt>.npy`, `time_diff_stds_dt<dt>.npy` (per-channel deltas at step `dt`)

Run, here on 8 ranks; without `mpirun -n 8` it runs serially, see [Running the scripts](#running-the-scripts). It works on a VDS or a directory of yearly `.h5`:
```
mpirun -n 8 python -m makani.datasets.compute_stats \
  --input_path /mnt/era5/makani_era5_train_vds.h5v \
  --metadata_file /mnt/era5/metadata.json \
  --output_path /mnt/era5/stats/train \
  --dt 1 \
  --batch_size 16 \
  --reduction_group_size 8 \
  --wind_angle_aware \
  --fail_on_nan
```

Key flags:
- `--input_path`: a folder of yearly `.h5` files or a single VDS `.h5v`.
- `--metadata_file`: matches the dataset/VDS (channels/lat/lon/dhours must align).
- `--output_path`: directory where the `.npy` outputs are written.
- `--dt`: time-step (in units of `dhours`) for difference stats.
- `--wind_angle_aware`: treat paired wind components as magnitudes for mean/std. This conserves the angle of the wind vector and just normalizes its magnitudes.
- `--wind_stats`: how the paired wind components (`u500`/`v500`, ...) are normalized: `componentwise` (default) like every other channel; `magnitude` is the same as `--wind_angle_aware`; `vector` treats each pair as a tangent vector field, with zero mean and one shared scale `sqrt(E[u^2 + v^2] / 2)`. A constant vector field is singular at the poles and not band-limited, so models that encode the wind as a tangent vector field need `vector`.
- `--fail_on_nan`: abort if NaNs appear instead of masking them. This should not be used when computing stats for fields with missing numbers such as `sst`.
- `--batch_size`: samples per read; tune for memory.
- `--reduction_group_size`: size of the groups of the parallel reduction (performance tuning). It is clamped to the largest divisor of the number of ranks that does not exceed it, so the default of 8 works on any number of ranks, also on one.

The spatial averages take the quadrature weights of the dataset's grid, the `coords.grid_type` of its metadata file.

### Time invariant fields

`convert_era5_invariants.py` is the counterpart of
`convert_era5.py` for fields without a time axis. It takes the same
subcommands (`wb2` or `ncar`) and the same metadata json, of which only the grid is used,
so the metadata of the dataset the invariants belong to can be passed as is. The fields
are selected with `--channels` by ECMWF short name and default to all of them:

| name | field | name | field |
|---|---|---|---|
| `z` | surface geopotential (orography) | `tvl`, `tvh` | type of low / high vegetation |
| `lsm` | land-sea mask | `sdor` | standard deviation of orography |
| `slt` | soil type | `isor` | anisotropy of sub-gridscale orography |
| `cl` | lake cover | `anor` | angle of sub-gridscale orography |
| `dl` | lake depth | `slor` | slope of sub-gridscale orography |
| `cvl`, `cvh` | low / high vegetation cover | `sdfor` | standard deviation of filtered subgrid orography |

The invariants are a few megabytes in total, so the script runs serially and writes a
single file:
```
python -m makani.datasets.convert_era5_invariants ncar \
  --output_file "/path/to/output/invariants.h5" \
  --metadata_file "/path/to/metadata.json" \
  --channels z lsm slt cvl cvh tvl tvh
```
The file has the layout of a yearly file without the time axis: `fields` is
`(channel, lat, lon)`, `valid_data` is `(channel,)`, and there is no `timestamp`. The
`wb2` source takes `--input_file` and `--coord_mode` as for the time dependent data; it
reads the invariants at the first timestep where a store repeats them along time, as
ARCO-ERA5 does. Not every store holds every field (the WeatherBench2 13 level stores, for
instance, lack `dl`), and `--skip_missing_channels` writes those as missing rather than
failing. `--force_overwrite` replaces an existing file.

Training selects channels from this file and says how each is encoded, see
`makani/data/invariants.py`:
```yaml
invariants_path: /invariants/invariants.h5
invariants:
  - {channel: z, encoding: normalize}
  - {channel: lsm, encoding: onehot, num_classes: 2, rounding: floor}
```
