# Changelog

## 0.3.0

Changes since 0.3.0rc1.

**Behavior changes**
- Bias, spectrum and zonal-spectrum files give lead times in hours, like the rollout and metrics files, and the zonal spectrum's Nyquist bin is half its previous value (#198).
- Pangu checkpoints trained before the shifted-window roll fix do not reproduce exactly (#198).
- An unknown `scheduler` name raises instead of silently training without one (#198).
- The ACC, SSR and spread of data with missing values come out lower than before, now that each sample counts once; they were too large by the inverse of the valid-area fraction (its square root for the spread) (#202).
- moviepy is no longer needed: the visualizer writes its GIFs with Pillow, `make_rollout_video.py` writes with imageio/imageio-ffmpeg, and the `vis` extra installs those. matplotlib is a core dependency (#207, #208).

**Fixes**
- Metrics: ACC, SSR and spread of masked data divided by the valid area twice (#202). Climatology and mask lookup with history or across leap years, with a clear error for times out of range (#201).
- Losses: `UnboundLocalError` in `CRPSLoss`, `GradientCRPSLoss`, `VortDivCRPSLoss` and `KernelScoreLoss` for single-member forecasts with spatial weights (#204). Gauss CRPS with `ensemble_weights`, and `GradientCRPSLoss(absolute=False)` channel weights (#198).
- Spectrum buffers counted the Nyquist order twice; `load_time_loop` used `dt` instead of `dt * dhours` hours; `BandLimitMask` ignored its default band limit; Pangu's shifted-window roll was displaced in longitude; remaining CUDA-only calls are guarded for CPU runs (#198).
- Constraint handling for water channels cleaned up in FCN3 and the constraints wrapper (#200).

**Tooling and CI**
- Rollout plotting script, and a sync after inference (#203). Docker images build for B300/GB300 (`10.3a`) (#206).
- CI runs DALI tests (#197), and broad new coverage of inference, model packages, training resume, schedulers and configs (#198).
- Distributed tests skip ensemble cases whose ensemble does not split evenly (#205), print from rank 0 only under srun/mpirun (#209), and skip the transformer-engine cases without CUDA (#210).

## 0.3.0rc1

**Breaking**
- Python >= 3.11, torch >= 2.12.1, torch-harmonics >= 0.9.0, PhysicsNeMo >= 1.3.0 (1.x and 2.x), DALI 2.x (`nvidia-dali-cuda120`), zarr >= 3. Dropped numba, pynvml, progressbar2, jsbeautifier. New `test`, `data_process`, `legacy_data_process` extras.
- Dataloaders emit dict batches (`input`, `target`, `*_dynamic_features`, `*_timestamp`; `(B, T, C, *grid)`). Timestamps are always present, and `return_timestamp` is gone.
- `Rollout` owns the multistep loop, and `StepWrapper` replaces `SingleStepWrapper`/`MultiStepWrapper`. The preprocessor is stateless between steps. `multistep` in `get_model`/`load_model_package` has no effect and warns, and `get_model` no longer accepts other extra arguments. `continue_noise` is replaced by `Rollout(noise_state=...)`.
- DALI dataloader state format 2 (timestamps are pipeline outputs, on the GPU): a dataloader state saved before is refused on resume, or skipped with `strict_restore: False`.
- Old DALI helpers (`data_loader_dali_2d`, `dali_es_helper_*`) were replaced by `dali_dataloader` + `sample_source`. Removed `img_utils`, `factorizations`, `drift_regularization`, `DistributedAFNO2Dv2`, and the fin/fout matmul split, which is now plain fork-join.
- SFNO: no MLP output bias before instance norms (dropped on load); the frequency `pos_embed` has a full `m = 0` column. Old checkpoints of the latter need conversion, and runs from before cannot resume their optimizer state.
- Invariants are stored in a single file; the model package renamed `land_sea_mask.nc` to `land_mask.nc` and normalizes its file names. Cosine zenith calculation uses an optimized vectorized routine.

**Data**
- Pluggable storage backends: makani HDF5, concatenated HDF5, makani zarr, ARCO/WB2 zarr, and native ICON mesh. Datasets declare and verify their grid. Shared channel mapping and metadata rework.
- Subsampling/resampling, NaN targets with masked losses and imputation, leap-year-safe concatenation, dataloader save/restore, GDS and O_DIRECT IO.
- Conservative remapping (raster and mesh to lat/lon), serial and spatially distributed.
- Converters: WB2 and NSF NCAR ERA5 merged into one tool with pluggable sources, NCAR downloader, h5_convert fixes, improved stats (Welford, power spectra).

**Models**
- Model package rework built on `Rollout`.
- Stochastic MLP with log-std spread, and a reworked noise module that is compilable and has a save/restore state.

**Training**
- Multistep weighting modes and multistep checkpointing. The ensemble is folded into the batch dimension. Validation streams one lead time at a time. Trainers no longer read a setup batch, so the first DALI epoch is complete.
- Losses: MMD, kernel score, corrected and memory-efficient energy score, and relative losses. Fixes to CRPS gradients, the coherence loss, and the energy-score spread at E=1. Losses are AMP-safe.
- `torch.compile` and inductor support (step-level compilation), FP8/TE autocast options, weight-decay param groups, gradient-norm tracking, and an LR warmup start.
- Checkpoints carry a parameter-name checksum. Fixes to the distributed instance norm and to the parallel comm routines, which moved from PhysicsNeMo to TH/makani.

**Inference and tooling**
- Streaming rollout buffer (GDS, O_DIRECT), WB2-compliant output, and an inferencer built on `Rollout`. With `n_history > 0`, the buffer records the latest step of the window, at its time, as the initial condition, not the oldest.
- Visualization rewrite: op registry instead of eval, rank 0 only, progress bars on videos.
- Benchmark tool, profiling/NVTX fixes, pre-commit and a style CI, CI on pull requests, API docs, SECURITY.md, and broad test coverage, including distributed tests.
