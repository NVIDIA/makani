# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from typing import Optional
import sys
import shutil
import glob
import argparse as ap 
import h5py as h5
from tqdm import tqdm


def h5_convert(input_dir: str,
               output_dir: str,
               chunksize: str,
               compression_mode: str,
               compression_parameter: Optional[int]=None,
               batchsize: Optional[int]=100,
               transpose: Optional[bool]=False,
               overwrite: Optional[bool]=False,
               entry_key: Optional[str]="fields"):

    """Function to reformat HDF5 dataset, e.g. enabling chunking or compression.
    
    ...

    Parameters
    ----------
    input_dir : str
        directory to where the dataset files are located which are to be converted.
    output_dor : str
        directory to where the converted files will be written to. The filename will be the same as the unconverted files.
    chunksize : str
        Chunksize specified as string for the HDF5 dataset in the new file, The following values are supported:
        none: no chunking, the whole dataset is in a single chunk
        auto: use HDF5 automatic chunking. 
        <some number>MB: set chunk size to <some number> megabytes
        (chunk_0, chunk_1, chunk_2, chunk_3): set chunk size independently for the individual dimensions.
    compression_mode : str
        Compression mode for the HDF5 dataset. Supported values are:
        none: no compression
        lzf: use LZF compression
        gzip: use GZIP compression
        szip: use SZIP compressions
        scaleoffset: use lossy scale-offset compression
    compression_parameter : int
        Parameter for the compression algorithm. Only supported for gzip or scaleoffset.
    batchsize : int
        Batch size for processing the dataset, performance options.
    transpose : bool
        Flag to transpose the data from channels first to channels last (NCHW to NHWC).
    overwrite : bool
        Flag to overwrite existing output files.
    entry_key: str
        This is the HDF5 dataset name of the data in the files. Defaults to "fields".
    """
    
    # set chunksize
    if chunksize == "auto":
        chunksize = True
        print("Setting chunksize to auto")
    elif chunksize == "none":
        chunksize = None
        print("Setting chunksize to none")
    elif "MB" in chunksize:
        chunksize = int(chunksize.replace("MB", "")) * 1024 * 1024
        print(f"Setting chunksize to {chunksize} MB")
    elif len(chunksize.split(",")) > 1:
        chunksize = tuple([int(x) for x in chunksize.split(",")])
        print(f"Setting chunksize to {chunksize}")
    else:
        raise ValueError(f"Error, chunksize {chunksize} not supported.")
    
    # get files
    files = glob.glob(os.path.join(input_dir, "*.h5"))

    # assemble kwargs
    kwargs = dict(chunks=chunksize)
    if compression_mode == "szip":
        kwargs["compression"] = "szip"
    elif compression_mode == "lzf":
        kwargs["compression"] = "lzf"
    elif compression_mode == "gzip":
        kwargs["compression"] =	"gzip"
        if compression_parameter is not None:
            kwargs["compression_opts"] = compression_parameter
    elif compression_mode == "scaleoffset":
        if (compression_parameter is not None) and (compression_parameter >= 0):
            kwargs["scaleoffset"] = compression_parameter

    # loop over files
    for ifname in files:

        #construct output file name
        ofname = os.path.join(output_dir, os.path.basename(ifname))

        # check if output file exists
        if os.path.exists(ofname):
            if not overwrite:
                print(f"File {ofname} already exists, skipping.", flush=True)
                continue
            else:
                os.remove(ofname)

        print(f"Converting {ifname} -> {ofname}", flush=True)
        with h5.File(ifname, 'r') as fin:

            # input data handle
            data_handle = fin[entry_key]

            # get dimension scales
            timestamps = fin["timestamp"][...]
            channel_names = fin["channel_names"][...]
            channel_names = [c.decode("ascii").strip() for c in channel_names.tolist()]
            chanlen = max([len(v) for v in channel_names])
            lat = fin["lat"]
            lon = fin["lon"]

            with h5.File(ofname, 'w') as fout:

                # output dataset
                fout.create_dataset(entry_key, data_handle.shape, dtype=data_handle.dtype, **kwargs)

                # dimension scales
                fout.create_dataset("timestamp", data=timestamps)
                fout.create_dataset("channel", len(channel_names), dtype=h5.string_dtype(length=chanlen))
                fout["channel"][...] = channel_names
		fout.create_dataset("lat", data=lat)
                fout.create_dataset("lon", data=lon)

                # create scales
                fout["timestamp"].make_scale("timestamp")
                fout["channel"].make_scale("channel")
                fout["lat"].make_scale("lat")
                fout["lon"].make_scale("lon")

                # label dimensions
                fout[entry_key].dims[0].label = fin[entry_key].dims[0].label
                fout[entry_key].dims[1].label = fin[entry_key].dims[1].label
                fout[entry_key].dims[2].label = fin[entry_key].dims[2].label
                fout[entry_key].dims[3].label = fin[entry_key].dims[3].label

                # attach scales
                fout[entry_key].dims[0].attach_scale(f["timestamp"])
                fout[entry_key].dims[1].attach_scale(f["channel"])
                fout[entry_key].dims[2].attach_scale(f["lat"])
                fout[entry_key].dims[3].attach_scale(f["lon"])

                # write data in batched fashion
                for start in tqdm(range(0, data_handle.shape[0], batchsize)):
                    end = min(start+batchsize, data_handle.shape[0])
                    data = data_handle[start:end, ...]

                    if transpose:
                        data = np.transpose(data, (0, 2, 3, 1))
                    
                    # write data
                    fout[entry_key][start:end, ...] = data[...]

    return


def main(args):
    h5_convert(input_dir=args.input_dir,
	       output_dir=args.output_dir,
               chunksize=args.chunksize,
	       compression_mode=args.compression_mode,
	       compression_parameter=args.compression_parameter,
               batchsize=args.batchsize,
               transpose=args.transpose,
               overwriteargs.overwrite)

    return


if __name__ == "__main__":
    # argparse
    parser = ap.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Directory with input files.", required=True)
    parser.add_argument("--output_dir", type=str, help="Directory for output files.", required=True)
    parser.add_argument("--chunksize", type=str, default="auto", help="Default chunksize.")
    parser.add_argument("--batchsize", type=int, default=100, help="Batch size for IO.")
    parser.add_argument("--scaleoffset", type=int, default=-1, help="Value for scaleoffset filter, negative values disable the filter..")
    parser.add_argument("--compression_mode", type=str, default=None, choices=["gzip", "szip", "scaleoffset", "lzf"], help="Which compression mode to use.")
    parser.add_argument("--compression_parameter", type=int, default=None, help="Value for compression filters, ignored when compression_mode is None")
    parser.add_argument("--transpose", action='store_true')
    parser.add_argument("--overwrite", action='store_true')
    args = parser.parse_args()

    main(args)
        
