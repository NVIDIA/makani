# Data Process

This folder contains python files for processing the data.

## Overview of processing

ERA5 data is provided in .h5 file format. Often data for several years and several variables is split up into files for each respective year, then containing the data for all variables, all timepoints during that year and all spatial locations. Each h5 file represents a dataset that contains additional metadata relevant for the processing of the data.

### Directory structure
This folder is organized as follows:

```
makani
├── ...
├── data_process                         # code related to pre-processing the data
│   ├── annotate_dataset.py              # annotation of the dataset
│   ├── concatenate_dataset.py           # concatenation of data files across several years
│   ├── convert_makani_output_to_wb2.py  # converting makani output to wb2 format
│   ├── convert_wb2_to_makani_input.py   # convert wb2 input in makani format
│   ├── data_process_helpers.py          # helper functions for distributed Welford reductions
│   ├── generate_wb2_climatology.py      # generate mask and dataset for climatology data
│   ├── get_histograms.py                # compute histograms for each variable over a dataset
│   ├── get_stats.py                     # compute power spectra for each variable over a dataset
│   ├── get_stats.py                     # calculate stats from the dataset
│   ├── h5_convert.py                    # reformat h5 files to enable compression/chunking
│   ├── merge_wb2_dataset.py             # add additional fields to an existing makani dataset from the Weatherbench dataset repo
│   ├── postprocess_stats.py             # postprocessg of stats
│   ├── wb2_helpers.py                   # wb2 helper functions
│   └── Readme.md                        # this file
...

```

### Annotate dataset

For scoring, the .h5 files are expected to be annotated with the correct metadata and dates. `annotate_dataset.py` modifies the files such that all relevant metadata is contained in the dataset, and that these metadata is universally equal across different dataset. Here, the original data is read, and for each file, metadata timestamps, latitude, longitude and channel are copied from the original file. Labels are provided in a uniform way. Timestamps are edited by converting the start sample into UTC time zone and deriving all following time samples from the UTC converted one. This ensure a unform time information across datasets.

### Concatenate dataset

`concatenate_dataset.py` creates a virtual dataset by combining several .h5 files into a single (virtual) dataset. This virtual dataset represents data from several years.

### Compute statistics and histograms

`get_stats.py ` several statistics calculated for either a folder containing several h5 files, or a combined h5f dataset (virtual dataset, concatenated across several years). Calculated stats are: global_means, global_stds, mins, maxs, time-means, time_diff_means_dt, time_diff_stds_dt. In a similar fashion, `get_histograms.py` computes histograms from the dataset

### Weatherbench

Makani contains several files to enable scoring consistent with Weatherbench2. `generate_wb2_climatology.py` computes climatology data provided by WeatherBench2 (ERA5 data, averaged data from 1990 - 2019) and converts them to a h5 dataset. Additionally, generates a climatology masks used by WB. Other helper functions are contained in `wb2_helpers.py`. `convert_wb2_to_makani_input` can be used to convert Weatherbench2 data such as the ARCO-ERA5 dataset to a makani-compatible format. `convert_makani_output_to_wb` converts makani inference output to Weatherbench2.

## Data Processing Examples

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
2. Run the converter (MPI optional but recommended for speed):
   ```
   mpirun -n 8 python convert_wb2_to_makani_input.py \
     --input_file "gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2" \
     --output_dir "/path/to/output/makani_era5" \
     --metadata_file "/path/to/metadata.json" \
     --years 2018 2019 \
     --batch_size 8 \
     --skip_missing_channels \
     --impute_missing_timestamps
   ```
   Flags you may want to use:
   - `--force_overwrite` to replace existing yearly files.
   - Omit `--skip_missing_channels` / `--impute_missing_timestamps` to fail fast on gaps.

The command writes one Makani-format `.h5` file per requested year in `output_dir`
with shared dimension scales (`timestamp`, `channel`, `lat`, `lon`) and a
`valid_data` mask that tracks imputed or missing values.

### Concatenate individual years into a single dataset

The script `concatenate_dataset.py` builds a virtual HDF5 dataset (VDS) that references the existing yearly Makani `.h5` files; no data is copied. The VDS stores the exact paths to the source files, so those paths must remain readable later (e.g.,
training or inference). If you run inside Docker, mount the yearly files at the
same absolute locations that were used when the VDS was created.

Basic usage (single directory of yearly files):
```
python concatenate_dataset.py \
  --dataset_metadata /data/makani_era5/metadata.json \
  --input_dirs /data/makani_era5 \
  --output_file /data/makani_era5/makani_era5_vds.h5 \
  --dhours_rel 1          # 1 keeps every timestep; a value >1 employs temporal subsampling
```

Path consistency with Docker:
```
# Create VDS on the host (absolute paths are embedded in the VDS):
python concatenate_dataset.py \
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

The script `get_stats.py` computes per-channel statistics for a Makani dataset (yearly files or a virtual HDF5 created via `concatenate_dataset.py`). Outputs are NumPy files:
- `global_means.npy`, `global_stds.npy` (per-channel scalars)
- `mins.npy`, `maxs.npy` (per-channel scalars)
- `time_means.npy` (per-channel spatial climatology)
- `time_diff_means_dt<dt>.npy`, `time_diff_stds_dt<dt>.npy` (per-channel deltas at step `dt`)

Run (MPI recommended for speed; works on VDS or a directory of yearly `.h5`):
```
mpirun -n 8 python get_stats.py \
  --input_path /mnt/era5/makani_era5_train_vds.h5v \   # or a folder with .h5 files
  --metadata_file /mnt/era5/metadata.json \
  --output_path /mnt/era5/stats/train \
  --dt 1 \
  --quadrature_rule gauss-legendre \
  --batch_size 16 \
  --reduction_group_size 8 \
  --wind_angle_aware \
  --fail_on_nan
```

Key flags:
- `--input_path`: folder of yearly `.h5` files or a single VDS `.h5v`.
- `--metadata_file`: matches the dataset/VDS (channels/lat/lon/dhours must align).
- `--output_path`: directory where the `.npy` outputs are written.
- `--dt`: time-step (in units of `dhours`) for difference stats.
- `--quadrature_rule`: spatial weighting (`naive`, `clenshaw-curtiss`, `gauss-legendre`).
- `--wind_angle_aware`: treat paired wind components as magnitudes for mean/std. This conserves the angle of the wind vector and just normalizes its magnitudes.
- `--fail_on_nan`: abort if NaNs appear instead of masking them. This should not be used when computing stats for fields with missing numbers such as `sst`.
- `--batch_size`: samples per read; tune for memory.
- `--reduction_group_size`: MPI all-reduce group size (performance tuning).