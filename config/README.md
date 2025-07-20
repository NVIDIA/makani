## Model recipes / Training configurations

This folder contains the configurations for training the ML models. This folder is structured as follows:


```
makani
├── ...
├── config                  # configurations
│   ├── afnonet.yaml        # baseline configurations from original FourCastNet paper
│   ├── icml_models.yaml    # SFNO baselines from the ICML paper
│   ├── sfnonet.yaml        # stable SFNO baselines
│   ├── vit.yaml            # baseline configurations for a Vision Transformer
│   └── Readme.md           # this file
...

```

Baselines from the original FourCastNet paper can be found in `afnonet.yaml`. Configurations for Spherical FNOs from the ICML publication can be found in `icml_models.yaml`. The latest and greatest SFNO configurations are found in `sfnonet.yaml`.