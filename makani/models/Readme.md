## Networks

This folder contains models for training and associated code. Models that are currently supported can be queried by calling `makani.models.model_registry.list_models()`.

### Directory structure
This folder is organized as follows:

```
makani
├── ...
├── models                          # code realted to ML models
│   ├── common                      # folder containing common features used in the neworks
│   │   ├── activations.py          # complex activation functions
│   │   ├── bandlimit.py            # band limiting on the sphere
│   │   ├── contractions.py         # einsum wrappers for complex contractions
│   │   ├── distributed             # the distributed versions of the layers: matmul, MLP, patch embedding, attention, norms, FFTs
│   │   ├── fft.py                  # real FFTs with mode truncation
│   │   ├── imputation.py           # learned imputation of missing values
│   │   ├── layer_norm.py           # layer and instance norm kernels
│   │   ├── mlp.py                  # encoder/decoder and MLP
│   │   ├── norm_layers.py          # the normalization layer of a network, picked by name, serial or distributed
│   │   ├── patch.py                # patch embeddings and recoveries
│   │   ├── pos_embedding.py        # position embeddings
│   │   ├── regularization.py       # drop path, seeded dropout, layer scale
│   │   ├── resample.py             # up- and downsampling
│   │   └── spectral_convolution.py # spectral convolution layers for (S)FNO architectures
│   ├── networks                    # contains the actual architectures
│   │   ├── afnonet_v2.py           # optimized AFNO
│   │   ├── afnonet.py              # AFNO implementation
│   │   ├── debug.py                # dummy network for debugging purposes
│   │   ├── fourcastnet3.py         # FourCastNet 3
│   │   ├── fourcastnet3_1.py       # FourCastNet 3.1
│   │   ├── pangu.py                # implementation of Pangu-Weather
│   │   ├── sfnonet.py              # implementation of (S)FNO
│   │   ├── snonet.py               # spherical neural operator (FourCastNet 2)
│   │   └── vit.py                  # implementation of a VIT
│   ├── constants.py                # physical constants
│   ├── constraints.py              # constraints on the output channels of a network
│   ├── helpers.py                  # helper functions
│   ├── model_input.py              # ModelInput, the input of a network, and FlatInputAdapter: input noise and packing for flat networks
│   ├── model_registry.py           # model registry with get_model routine that takes care of wrapping the model
│   ├── noise                       # noise on the sphere
│   │   ├── processes.py            # noise processes: white, diffusion and dummy noise
│   │   └── controller.py           # NoiseController: advances, snapshots, restores and re-seeds the noise of a model
│   ├── parametrizations.py         # output parametrizations such as hydrostatic balance
│   ├── physicsnemo_compat.py       # building PhysicsNeMo modules across 1.x and 2.x
│   ├── preprocessor.py             # stateless preprocessor: history normalization, dynamic and static features, bias correction
│   ├── preprocessor_helpers.py     # bias correction and other inputs of the preprocessor
│   ├── stepper.py                  # StepWrapper: binds a network to its preprocessor and its input, runs a Rollout
│   ├── stochastic_interpolant.py   # forecasting with stochastic interpolants
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

### Grids

A network works on the grid of its data, which the parameters name: `model_grid_type`, and for the spherical networks `sht_grid_type`, the grid of their processor. A grid type is a regular grid of torch-harmonics, `equiangular`, `legendre-gauss`, `lobatto`, `fejer1` or `trapezoidal`, or `euclidean` for a field in the plane. The data is moved from the grid its metadata declares, `data_grid_type`, onto `model_grid_type` by the dataloader (`makani.utils.grids.GridConverter`).

`get_model` passes the shapes of the input and the output, `inp_shape` and `out_shape`, and the grid types among the other parameters, so the constructors of the networks take shapes and names: PhysicsNeMo stores the constructor arguments of a model in its checkpoints. Inside, a network builds its grids once and hands the grid descriptors to its layers, rather than shapes and names:

```python
from makani.utils.grids import make_grid, sht_grid

self.inp_grid = sht_grid(inp_shape, model_grid_type)  # make_grid also takes the plane
self.out_grid = sht_grid(out_shape, model_grid_type)
self.internal_grid = sht_grid((h, w), sht_grid_type)
```

The spectral transforms, the DISCO convolutions, the norms (`get_norm_layer_handle`), the position embeddings and `BandLimitMask` take these grids, and so do the losses, the metrics and the noise processes, which their handlers build from the same parameters. `spherical_bandlimit(grid)` gives the band limit the spectral layers truncate to.

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
from makani.data.rollout import Rollout

rollout = Rollout(batch, noise=model.noise)  # advances the noise before each step
preds = model(rollout)

model.noise.set_rng(seed=333)                # reproduce, or decorrelate ensemble members
snapshot = model.noise.snapshot()            # rewind point
model.noise.restore(snapshot)                # rewind
```

A model package exposes the same controller as `ModelWrapper.noise`. A rollout of a stochastic model has to drive the noise of that model, and rollout checkpointing (`multistep_checkpoint`) draws the noise of each step again from the state of that step when it recomputes the step in the backward pass, after the rollout has advanced the noise.

### Model packages

Model packages are used for seamless inference outside of this repository. They define a flexible interface which takes care of normalization, dynamic features such as the cosine of the solar zenith angle, static features etc. Model packages are how [Earth2Studio](https://github.com/NVIDIA/earth2studio) runs makani models.
