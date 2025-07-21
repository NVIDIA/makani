# Makani: Massively parallel training of machine-learning based weather and climate models

[**Overview**](#overview) | [**Getting started**](#getting-started) | [**More information**](#more-about-makani) | [**Contributing**](#contributing) | [**Further reading**](#further-reading) | [**References**](#references)

[![tests](https://github.com/NVIDIA/makani/actions/workflows/tests.yml/badge.svg)](https://github.com/NVIDIA/makani/actions/workflows/tests.yml)

Makani (the Hawaiian word for wind 🍃🌺) is a library designed to enable the research and development of the next generation of machine-learning (ML) based weather and climate models in PyTorch. Makani was used to train [FourCastNet3 [1]](https://arxiv.org/abs/2507.12144v2), [Spherical Fourier Neural Operators (SFNO) [2]](https://arxiv.org/abs/2306.03838) for weather (FourCastNet2), [Huge ensemble of SFNO (HENS-SFNO) [3,4]](https://arxiv.org/abs/2408.03100), and [FourCastNet1 [5]](https://arxiv.org/abs/2202.11214).

Makani is aimed at researchers working on ML based weather prediction. Stable features are frequently ported to the [earth2studio](https://github.com/NVIDIA/earth2studio) and the [NVIDIA PhysicsNeMo](https://developer.nvidia.com/physicsnemo) framework. For commercial and production purposes, we recommend checking out these packages.

<div align="center">
<img src="https://github.com/NVIDIA/makani/blob/main/images/fcn3_ens3_640p"  height="388px">
</div>

## Overview

Makani is a research code developed by engineers and researchers at NVIDIA and NERSC for massively parallel training of weather and climate prediction models on 100+ GPUs and to enable the development of the next generation of weather and climate models. Makani is written in [PyTorch](https://pytorch.org) and supports various forms of model- and data-parallelism, asynchronous loading of data, unpredicted channels, autoregressive training and much more. Makani is fully configurable through .yaml configuration files and support flexible development of novel models. Metrics, Losses and other components are designed in a modular fashion to support configurable, custom training- and inference-recipes at scale. Makani also supports scalable, fully online scoring modes, which are compatible with WeatherBench2. Among others, Makani was used to train the [FourCastNet](https://research.nvidia.com/publication/2025-07_fourcastnet-3-geometric-approach-probabilistic-machine-learning-weather) models, on the ERA5 dataset.

## Getting started

Makani can be installed by running

```bash
git clone git@github.com:NVIDIA/makani.git
cd makani
pip install -e .
```

### Training:

Makani supports ensemble and deterministic training. Ensemble training is launched by calling `ensemble.py`, whereas deterministic training is launched by calling `train.py`. Both scripts expect the CLI arguments to specify the configuration file `--yaml_config` and he configuration target `--config`, which is contained in the configuration file:

```bash
mpirun -np 8 --allow-run-as-root python -u train.py --yaml_config="config/fourcastnet3.yaml" --config="fcn3_sc2_edim45_layers10_pretrain1"
```

Makani supports various optimization to fit large models ino GPU memory and enable computationally efficient training. An overview of these features and corresponding CLI arguments is provided in the following table:

| Feature                   | CLI argument                                  | options                      |
|---------------------------|-----------------------------------------------|------------------------------|
| Batch size                | `--batch_size`                                | 1,2,3,...                    |
| Ensemble size             | `--ensemble_size`                             | 1,2,3,...                    |
| Automatic Mixed Precision | `--amp_mode`                                  | `none`, `fp16`, `bf16`       |
| Just-in-time compilation  | `--jit_mode`                                  | `none`, `script`, `inductor` |
| Activation checkpointing  | `--checkpointing_level`                       | 0,1,2,3                      |
| Channel parallelism       | `--fin_parallel_size`, `--fout_parallel_size` | 1,2,3,...                    |
| Spatial model parallelism | `--h_parallel_size`, `--w_parallel_size`      | 1,2,3,...                    |
| Ensemble parallelism      | `--ensemble_parallel_size`                    | 1,2,3,...                    |
| Multistep training        | `--multistep_count`                           | 1,2,3,...                    |
| Skip training             | `--skip_training`                             |                              |
| Skip validation           | `--skip_validation`                           |                              |

Especially larger models are enabled by using a mix of these techniques. Spatial model parallelism splits both the model and the data onto multiple GPUs, thus reducing both the memory footprint of the model and the load on the IO as each rank only needs to read a fraction of the data. A typical "large" training run of SFNO can be launched by running

```bash
mpirun -np 256 --allow-run-as-root python -u makani.train --amp_mode=bf16 --multistep_count=1 --run_num="ngpu256_sp4" --yaml_config="config/sfnonet.yaml" --config="sfno_linear_73chq_sc3_layers8_edim384_asgl2" --h_parallel_size=4 --w_parallel_size=1 --batch_size=64
```
Here we train the model on 256 GPUs, split horizontally across 4 ranks with a batch size of 64, which amounts to a local batch size of 1/4. Memory requirements are further reduced by the use of `bf16` automatic mixed precision.

### Inference:

Makani supports scalable and flexible on-line inference aimed at minimizing data movement and disk I/O, which is well suited to the low inference costs of ML weather models and modern HPC infrastructure. In a similar fashion to training, inference can be called from the CLI by calling `inference.py` and handled by `inferencer.py`. To launch inference on the out-of-sample dataset, we can call:

```bash
mpirun -np 256 --allow-run-as-root python -u makani.inference --run_num="ngpu256_sp4" --yaml_config="config/sfnonet.yaml" --config="sfno_linear_73chq_sc3_layers8_edim384_asgl2" --batch_size=64
```

By default, the inference script will perform inference on the out-of-sample dataset and compute the mtrics. The inference script supports model, data and ensemble parallelism out of the box, enabling efficient and scalable scoring. The inference script support additional CLI arguments which enable validation on a subset of the dataset, as well as writing out inferred states:

| Feature                   | CLI argument                                  | options                      |
|---------------------------|-----------------------------------------------|------------------------------|
| Start date                | `--start_date`                                | 2018-01-01+UTC00:00:00       |
| End date                  | `--end_date`                                  | 2018-12-31+UTC24:00:00       |
| Date step (in hours)      | `--date_step`                                 | 1,2,...                      |
| Output file               | `--output_file`                               | file path for field outputs  |
| Output channels           | `--output_channels`                           | channels to write out        |
| Metrics file              | `--metrics_file`                              | file path for metrics output |
| Bias file                 | `--bias_file`                                 | file path for bias output    |
| Spectrum file             | `--spectrum_file`                             | file path for spectra output |

## More about Makani

### Project structure

The project is structured as follows:

```
makani
├── ...
├── config                      # configuration files, also known as recipes
├── data_process                # data pre-processing such as computation of statistics
├── datasets                    # dataset utility scripts
├── docker                      # scripts for building a docker image for training
├── makani                      # Main directory containing the package
│   ├── inference               # contains the inferencer
│   ├── mpu                     # utilities for model parallelism
│   ├── networks                # networks, contains definitions of various ML models
│   ├── third_party/climt       # third party modules
│   │   └── zenith_angle.py     # computation of zenith angle
│   ├── utils                   # utilities
│   │   ├── dataloaders         # contains various dataloaders
│   │   ├── metrics             # metrics folder contains routines for scoring and benchmarking.
│   │   ├── ...
│   │   ├── comm.py             # comms module for orthogonal communicator infrastructure
│   │   ├── dataloader.py       # dataloader interface
│   │   ├── metric.py           # centralized metrics handler
│   │   ├── trainer_profile.py  # copy of trainer.py used for profiling
│   │   └── trainer.py          # main file for handling training
│   ├── ...
│   ├── inference.py            # CLI script for launching inference
│   ├── train.py                # CLI script for launching training
├── tests                       # test files
└── README.md                   # this file
```

### Model and Training configuration
Model training in Makani is specified through the use of `.yaml` files located in the `config` folder. The corresponding models are located in `modelf` and registered in the model registry in `models/model_registry.py`. The following table lists the most important configuration options.

| Configuration Key         | Description                                             | Options                                                 |
|---------------------------|---------------------------------------------------------|---------------------------------------------------------|
| `nettype`                 | Network architecture.                                   | `FCN3`,`SFNO`, `SNO`, `AFNO`, `ViT`                     |
| `loss`                    | Loss function.                                          | `l2`, `geometric l2`, `amse`, `crps`...                 |
| `channel_weights`         | Weighting function for the respective channels.         | `constant`, `auto`, `uncertainty`...                    |
| `optimizer`               | Optimizer to be used.                                   | `Adam`, `AdamW`, `SGD`,`Sirfshampoo`                    |
| `lr`                      | Initial learning rate.                                  | float > 0.0                                             |
| `batch_size`              | Batch size.                                             | integer > 0                                             |
| `ensemble_size`           | Ensemble size.                                          | integer > 0                                             |
| `max_epochs`              | Number of epochs to train for                           | integer                                                 |
| `scheduler`               | Learning rate scheduler to be used.                     | `None`, `CosineAnnealing`, `ReduceLROnPlateau`, `StepLR`|
| `lr_warmup_steps`         | Number of warmup steps for the learning rate scheduler. | integer >= 0                                            |
| `weight_decay`            | Weight decay.                                           | float                                                   |
| `train_data_path`         | Directory path which contains the training data.        | string                                                  |
| `test_data_path`          | Network architecture.                                   | string                                                  |
| `exp_dir`                 | Directory path for ouputs such as model checkpoints.    | string                                                  |
| `metadata_json_path`      | Path to the metadata file `data.json`.                  | string                                                  |
| `channel_names`           | Channels to be used for training.                       | List[string]                                            |


For a more comprehensive overview, we suggest looking into existing `.yaml` configurations. More details about the available configurations can be found in [this file](config/README.md).

### Training data
Makani expects the training/test data in HDF5 format, where each file contains the data for an entire year. The dataloaders in Makani will then load the input `inp` and the target `tar`, which correspond to the state of the atmosphere at a given point in time and at a later time for the target. The time difference between input and target is determined by the parameter `dt`, which determines how many steps the two are apart. The physical time difference is determined by the temporal resolution `dhours` of the dataset.

Makani requires a metadata file named `data.json`, which describes important properties of the dataset such as the HDF5 variable name that contains the data. Another example are channels to load in the dataloader, which arespecified via channel names. The metadata file has the following structure:

```json
{
    "dataset_name": "give this dataset a name",     # name of the dataset
    "attrs": {                                      # optional attributes, can contain anything you want
        "decription": "description of the dataset",
        "location": "location of your dataset"
    },
    "h5_path": "fields",                            # variable name of the data inside the hdf5 file
    "dims": ["time", "channel", "lat", "lon"],      # dimensions of fields contained in the dataset
    "dhours": 6,                                    # temporal resolution in hours
    "coord": {                                      # coordinates and channel descriptions
        "grid_type": "equiangular",                 # type of grid used in dataset: currently suppported choices are 'equiangular' and 'legendre-gauss'
        "lat": [0.0, 0.1, ...],                     # latitudinal grid coordinates
        "lon": [0.0, 0.1, ...],                     # longitudinal grid coordinates
        "channel": ["t2m", "u10", "v10", ...]       # names of the channels contained in the dataset
    }
}
```

The ERA5 dataset can be downloaded [here](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview).

### Checkpoints and restarting

Makani supports 2 checkpointing formats `legacy` and `flexible`. By default, makani uses the `legacy` format, which saves the model, optimizer and scheduler state dicts into a file for each model parallel rank. This enables restoring the run in the same model-parallel configuration, including optimizer and scheduler states. Alternatively, if a different model-parallel configuration needs to be chosen, the `flexible` format can be used to restore the model in another configuration. Unfortunately, support for restoring optimizer states in this manner is still experimental. Which format is used for saving and restoring is determined by the

Makani offers a conversion script to convert `legacy` checkpoints into `flexible` checkpoints. It can be run by calling `python -u makani.convert_checkpoint` script.

### Model packages

By default, Makani will save out a model package when training starts. Model packages allow easily contain all the necessary data to run the model. This includes statistics used to normalize inputs and outputs, unpredicted static channels and even the code which appends celestial features such as the cosine of the solar zenith angle. Read more about model packages [here](networks/Readme.md).

## Contributing

Thanks for your interest in contributing. There are many ways to contribute to this project.

- If you find a bug, let us know and open an issue. Even better, if you feel like fixing it and making a pull-request, we are incredibly grateful for that. 🙏
- If you feel like adding a feature, we encourage you to discuss it with us first, so we can guide you on how to best achieve it.

While this is a research project, we aim to have functional unit tests with decent coverage. We kindly ask you to implement unit tests if you add a new feature and it can be tested.

## Further reading

- [FourCastNet 3 paper](https://arxiv.org/abs/2507.12144v2)
- [NVIDIA Research FCN3 site](https://research.nvidia.com/publication/2025-07_fourcastnet-3-geometric-approach-probabilistic-machine-learning-weather) on FourCastNet 3
- [NVIDIA blog article](https://developer.nvidia.com/blog/modeling-earths-atmosphere-with-spherical-fourier-neural-operators/) on Spherical Fourier Neural Operators for ML-based weather prediction
- [torch-harmonics](https://github.com/NVIDIA/torch-harmonics), a library for differentiable Spherical Harmonics in PyTorch
- [earth2studio](https://github.com/NVIDIA/earth2studio), a library for intercomparing DL based weather models
- [PhysicsNeMo](https://developer.nvidia.com/physicsnemo), NVIDIA's library for physics-ML
- [Dali](https://developer.nvidia.com/dali), NVIDIA data loading library

## Authors

<table>
  <tr>
    <td align="center" valign="middle">
      <img src="https://upload.wikimedia.org/wikipedia/commons/a/a4/NVIDIA_logo.svg" width="300px">
    </td>
    <td align="center" valign="middle">
      <img src="https://www.nersc.gov/_resources/themes/nersc/images/NERSC_logo_no_spacing.svg" height="100px">
    </td>
  </tr>
</table>

The code was developed by Thorsten Kurth, Boris Bonev, Ankur Mahesh, Dallas Foster, Jean Kossaifi, Animashree Anandkumar, Kamyar Azizzadenesheli, Noah Brenowitz, Ashesh Chattopadhyay, Yair Cohen, William D. Collins, Franziska Gerken, David Hall, Peter Harrington, Pedram Hassanzadeh, Christian Hundt, Karthik Kashinath, Zongyi Li, Morteza Mardani, Jaideep Pathak, Stefanos Pertigkiozoglou, Mike Pritchard, David Pruitt, Sanjeev Raja, Shashank Subramanian.


## References

<a id="#fcn3_paper">[1]</a>
Bonev B., Kurth T., Mahesh A., Bisson, M., Kossaifi J., Kashinath K., Anandkumar A. Collins W.D., Pritchard M., Keller A.;
FourCastNet 3: A geometric approach to probabilistic machine-learning weather forecasting at scale;
arXiv 2507.12144, 2025.

<a id="#sfno_paper">[2]</a>
Bonev B., Kurth T., Hundt C., Pathak, J., Baust M., Kashinath K., Anandkumar A.;
Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere;
arXiv 2306.0383, 2023.

<a id="#hens1_paper">[3]</a>
Mahesh A., Collins W.D., Bonev B., Brenowitz N., Cohen Y., Elms J., Harrington P., Kashinath K., Kurth T., North J., OBrian T., Pritchard M., Pruitt D., Risser M., Subramanian S., Willard J.
Huge Ensembles Part I: Design of Ensemble Weather Forecasts using Spherical Fourier Neural Operators;
arXiv 2408.03100, 2025.

<a id="#hens1_paper">[4]</a>
Mahesh A., Collins W.D., Bonev B., Brenowitz N., Cohen Y., Elms J., Harrington P., Kashinath K., Kurth T., North J., OBrian T., Pritchard M., Pruitt D., Risser M., Subramanian S., Willard J.
Huge Ensembles Part II: Properties of a Huge Ensemble of Hindcasts Generated with Spherical Fourier Neural Operators;
arXiv 2408.01581, 2025.

<a id="1">[5]</a>
Pathak J., Subramanian S., Harrington P., Raja S., Chattopadhyay A., Mardani M., Kurth T., Hall D., Li Z., Azizzadenesheli K., Hassanzadeh P., Kashinath K., Anandkumar A.;
FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators;
arXiv 2202.11214, 2022.

## Citation

If you use this package, please cite

```bibtex
@misc{bonev2025fourcastnet3,
      title={FourCastNet 3: A geometric approach to probabilistic machine-learning weather forecasting at scale},
      author={Boris Bonev and Thorsten Kurth and Ankur Mahesh and Mauro Bisson and Jean Kossaifi and Karthik Kashinath and Anima Anandkumar and William D. Collins and Michael S. Pritchard and Alexander Keller},
      year={2025},
      eprint={2507.12144},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.12144},
}
```
