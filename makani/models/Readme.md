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
│   ├── model_input.py              # ModelInput, the input of a network, and FlatInputAdapter: input noise and packing for flat networks
│   ├── model_package.py            # model package implementation
│   ├── model_registry.py           # model registry with get_model routine that takes care of wrapping the model
│   ├── noise.py                    # noise processes on the sphere: white, diffusion and dummy noise
│   ├── noise_controller.py         # NoiseController: advances, snapshots, restores and re-seeds the noise of a model
│   ├── preprocessor.py             # stateless preprocessor: history normalization, dynamic and static features, bias correction
│   ├── stepper.py                  # StepWrapper: binds a network to its preprocessor and its input, runs a Rollout
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

### The input of a network

In each step, the `StepWrapper` assembles the input of the network as a `ModelInput`: the state window `(B, n_history + 1, C, H, W)`, normalized with its history statistics, the dynamic features as the data holds them, and the static features, each in a field of its own. A network that declares `takes_model_input = True` gets it as it is:

```python
class MyNet(nn.Module):
    takes_model_input = True

    def forward(self, model_input):
        ...  # model_input.state, .stats, .dynamic_features, .static
```

Any other network is a flat network and gets a single channel stack, which the `FlatInputAdapter` of the wrapper packs: per time step `[state | dynamic features | noise]`, the dynamic features and the noise normalized with history statistics of their own, the time steps folded into the channels, then the static features.

### Stochastic models

The noise of a model is the model's own: the model owns its noise processes, so that their learnable parameters are trained and checkpointed with it, and draws from them where it uses them. Noise on the input is an `InputNoise` layer, which appends its draw as channels (`concatenate`) or adds it to selected channels of the state as a perturbation in data space (`perturb`); the `FlatInputAdapter` holds the one configured in `input_noise` for a flat network, and a network that takes a `ModelInput` builds its own, with `InputNoise.from_params(params)`. Noise in a latent space is drawn by the network wherever it needs it. Every process is listed through `noise_processes()`, and driven from outside only through `model.noise`, a `NoiseController`, which keeps its state consistent with the rollout:

```python
from makani.utils.dataloaders.rollout import Rollout

rollout = Rollout(batch, noise=model.noise)  # advances the noise before each step
preds = model(rollout)

model.noise.set_rng(seed=333)                # reproduce, or decorrelate ensemble members
snapshot = model.noise.snapshot()            # rewind point
model.noise.restore(snapshot)                # rewind
```

A model package exposes the same controller as `ModelWrapper.noise`. A rollout of a stochastic model has to drive the noise of that model, and rollout checkpointing (`multistep_checkpoint`) is refused for a network with noise processes of its own, since the recompute in the backward pass would read the noise after the rollout has advanced it.

### Model packages

Model packages are used for seamless inference outside of this repository. They define a flexible interface which takes care of normalization, dynamic features such as the cosine of the solar zenith angle, static features etc. Model packages seemlessly integrate with [earth2mip](https://github.com/NVIDIA/earth2mip).
