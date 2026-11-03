## Networks

This folder contains models for training and associated code. Models that are currently supported can be queried by calling `networks.models.list_models()`.

### Directory structure
This folder is organized as follows:

```
makani
├── ...
├── models                          # code realted to ML models
│   ├── common                      # folder containing common features used in the neworks
│   │   ├── activations.py          # complex activation functions
│   │   ├── contractions.py         # einsum wrappers for complex contractions
│   │   ├── layers.py               # common layers such as MLPs and wrappers for FFTs
│   │   └── spectral_convolution.py # spectral convolution layers for (S)FNO architectures
│   ├── networks                    # contains the actual architectures
│   │   ├── afnonet_v2.py           # optimized AFNO
│   │   ├── afnonet.py              # AFNO implementation
│   │   ├── debug.py                # dummy network for debugging purposes
│   │   ├── fourcastnet3.py         # FourCastNet 3
│   │   ├── fourcastnet3_1.py       # FourCastNet 3.1
│   │   ├── pangu.py                # implementation of Pangu-Weather
│   │   ├── pangu_onnx.py           # Pangu-Weather from its ONNX export
│   │   ├── sfnonet.py              # implementation of (S)FNO
│   │   ├── snonet.py               # spherical neural operator (FourCastNet 2)
│   │   └── vit.py                  # implementation of a VIT
│   ├── helpers.py                  # helper functions
│   ├── model_package.py            # model package implementation
│   ├── model_registry.py           # model registry with get_model routine that takes care of wrapping the model
│   ├── noise.py                    # noise processes on the sphere: white, diffusion and dummy noise
│   ├── noise_controller.py         # NoiseController: advances, snapshots, restores and re-seeds the noise of a model
│   ├── preprocessor.py             # stateless preprocessor: dynamic and static features, input noise, history normalization
│   ├── stepper.py                  # StepWrapper: binds a model to its preprocessor, runs a Rollout
│   └── Readme.md                   # this file
...

```

### Model registry

The model registry is a central place for organizing models in makani. By default, it contains the architectures contained in the `networks` directory, to which makani also exposes entrypoints. Models can be instantiated via

```python
from makani.models import model_registry

model = model_registry.get_model(params)
```

where `params` is the parameters object used to instantiate the model. Custom models can be registered in the registry using the `register` method. Models are required to take keyword arguments. These are automatically parsed from the `params` datastructure and passed to the model.

In addition, models can be automatically registered through the `nettype` field in the configuration yaml file. To do so, the user can specify

```yaml
nettype: "path/to/model_file.py:ModelName"
```

using the path to the model file and the class name `ModelName`.

### Stochastic models

The noise of a model lives in noise processes, modules the model owns, so that their learnable parameters are trained and checkpointed with it: the input noise of the preprocessor, and any process a network lists through a `noise_processes()` method, e.g. for noise in its latent space. They are driven through `model.noise`, a `NoiseController`, and never sampled by the caller:

```python
from makani.utils.dataloaders.rollout import Rollout

rollout = Rollout(batch, noise=model.noise)  # advances the noise before each step
preds = model(rollout)

model.noise.set_rng(seed=333)                 # reproduce, or decorrelate ensemble members
snapshot = model.noise.snapshot()            # rewind point
model.noise.restore(snapshot)                # rewind
```

A model package exposes the same controller as `ModelWrapper.noise`.

### Model packages

Model packages are used for seamless inference outside of this repository. They define a flexible interface which takes care of normalization, dynamic features such as the cosine of the solar zenith angle, static features etc. Model packages seemlessly integrate with [earth2mip](https://github.com/NVIDIA/earth2mip).
