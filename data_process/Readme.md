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
│   ├── generate_wb2_climatology.py      # generate mask and dataset for climatology data
│   ├── get_histograms.py                # get histograms
│   ├── get_stats.py                     # calculate stats from the dataset
│   ├── h5_convert.py                    # reformat h5 files to enable compression/chunking
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