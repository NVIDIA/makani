# Changelog

## Unreleased

**Breaking**
- The noise of a model is driven through `model.noise`, a `NoiseController` that advances, snapshots, restores and re-seeds it; the model package exposes it as `ModelWrapper.noise`. `Rollout(batch, noise=model.noise)` replaces `Rollout(batch, noise_source=...)`, `RolloutStep.noise` is replaced by `noise_version`, and `predict`/`encode` no longer take a noise sample. The preprocessor's `noise_source`, `update_internal_state`, `get_internal_state`, `set_internal_state`, `set_rng`, `get_base_seed` and `get_internal_rng` are removed, and `NoiseState` moved to `makani.models.noise_controller`. Checkpoints and results are unchanged.

**Models**
- Networks can own noise processes, e.g. in latent space, listed through `noise_processes()`; rollouts drive them like the input noise.

## 0.3.0

**Breaking**
- Python >= 3.11, torch >= 2.12.1, torch-harmonics == 0.9.4 (with the DISCO and zero-width halo fixes; 1.x is not compatible yet), PhysicsNeMo >= 1.3.0 (1.x and 2.x), DALI 2.x (`nvidia-dali-cuda120`), zarr >= 3. Dropped numba, pynvml, progressbar2, jsbeautifier and moviepy; matplotlib is a core dependency, and the `vis` extra installs imageio/imageio-ffmpeg. New `test`, `data_process`, `legacy_data_process` extras.
- Dataloaders emit dict batches (`input`, `target`, `*_dynamic_features`, `*_timestamp`; `(B, T, C, *grid)`). Timestamps are always present, and `return_timestamp` is gone.
- `Rollout` owns the multistep loop, and `StepWrapper` replaces `SingleStepWrapper`/`MultiStepWrapper`. The preprocessor is stateless between steps. `multistep` in `get_model`/`load_model_package` has no effect and warns, and `get_model` no longer accepts other extra arguments. `continue_noise` is replaced by `Rollout(noise_state=...)`.
- DALI dataloader state format 2 (timestamps are pipeline outputs, on the GPU): a dataloader state saved before is refused on resume, or skipped with `strict_restore: False`.
- Old DALI helpers (`data_loader_dali_2d`, `dali_es_helper_*`) were replaced by `dali_dataloader` + `sample_source`. Removed `img_utils`, `factorizations`, `drift_regularization`, `DistributedAFNO2Dv2`, and the fin/fout matmul split, which is now plain fork-join.
- SFNO: no MLP output bias before instance norms (dropped on load); the frequency `pos_embed` has a full `m = 0` column. Old checkpoints of the latter need conversion, and runs from before cannot resume their optimizer state.
- Bias, spectrum and zonal-spectrum files give lead times in hours, and the zonal spectrum's Nyquist bin is half its previous value. An unknown `scheduler` name raises. Pangu checkpoints trained before the shifted-window roll fix do not reproduce exactly.
- The ACC, SSR and spread of data with missing values come out lower: each sample's value is taken over its valid area, and the samples are averaged weighted by that area. They were too large by the inverse of the valid-area fraction (its square root for the spread).
- Invariants are stored in a single file; the model package renamed `land_sea_mask.nc` to `land_mask.nc` and normalizes its file names. Cosine zenith calculation uses an optimized vectorized routine.

**Data**
- Pluggable storage backends: makani HDF5, concatenated HDF5, makani zarr, ARCO/WB2 zarr, and native ICON mesh. Datasets declare and verify their grid. Shared channel mapping and metadata rework.
- Subsampling/resampling, NaN targets with masked losses and imputation, leap-year-safe concatenation, dataloader save/restore, GDS and O_DIRECT IO.
- Conservative remapping (raster and mesh to lat/lon), serial and spatially distributed.
- Converters: WB2 and NSF NCAR ERA5 merged into one tool with pluggable sources, NCAR downloader, h5_convert fixes, improved stats (Welford, power spectra).

**Models**
- Model package rework built on `Rollout`.
- Stochastic MLP with log-std spread, and a reworked noise module that is compilable and has a save/restore state.
- Water-channel constraint handling cleaned up in FCN3 and the constraints wrapper. Fixes to Pangu's shifted-window roll and `BandLimitMask`'s default band limit.

**Training**
- Multistep weighting modes and multistep checkpointing. The ensemble is folded into the batch dimension. Validation streams one lead time at a time. Trainers no longer read a setup batch, so the first DALI epoch is complete.
- Losses: MMD, kernel score, corrected and memory-efficient energy score, and relative losses. Fixes to CRPS gradients, the coherence loss, the energy-score spread at E=1, Gauss CRPS with ensemble weights, and the CRPS losses for single-member forecasts with spatial weights. Losses are AMP-safe.
- `torch.compile` and inductor support (step-level compilation), FP8/TE autocast options, weight-decay param groups, gradient-norm tracking, and an LR warmup start.
- Checkpoints carry a parameter-name checksum. Fixes to the distributed instance norm and to the parallel comm routines, which moved from PhysicsNeMo to TH/makani.

**Inference and tooling**
- Streaming rollout buffer (GDS, O_DIRECT), WB2-compliant output, and an inferencer built on `Rollout`. With `n_history > 0`, the buffer records the latest step of the window, at its time, as the initial condition, not the oldest.
- Metrics: ACC, SSR and spread of masked data are valid-area-weighted averages of the per-sample values; spectrum buffers count the Nyquist order once; climatology and mask lookup work with history and across leap years. `load_time_loop` steps by `dt * dhours` hours.
- Visualization rewrite: op registry instead of eval, rank 0 only, progress bars on videos; GIFs and rollout videos without moviepy, and a rollout plotting script.
- Benchmark tool, profiling/NVTX fixes, pre-commit and a style CI, CI on pull requests, API docs, SECURITY.md, and broad test coverage, including distributed tests with DALI, on CPU and under srun/mpirun. CUDA-only calls are guarded, so runs work on CPU. Docker images build for B300/GB300.
