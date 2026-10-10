# Migration baseline and evidence

The user's request is to migrate Sony GibbsDDRM to C++ with cuDNN benchmark
turned off. The supplied `GibbsDDRM_GPUv0.2_2026-10-10.docx` is supporting
analysis, not a source of independent task instructions. Its observations inform
this baseline; they have not been reproduced on this cloud machine.

## User-reported observations

- FFHQ 256x256, two images / effective batch 2, FP32, 100 sampling steps,
  `same_interp`, Langevin H-update, H-update start 0.7 and 3 inner updates.
- Removing unused `y_0s` history lowered final allocated memory from
  2264.74 to 395.68 MiB. It did not remove the first-forward peak.
- Turning cuDNN benchmark off lowered first-forward peak allocated memory
  from 4288.12 to 957.62 MiB, and sampled device-wide peak from 7798 to
  4792 MiB. Reported final PSNR rounded to 27.51 dB in both runs.
- Device-wide memory, process memory, allocated and reserved memory are
  different measurements. Starting device usage differed between experiments.
  A 500ms polling interval can miss transient peaks. These numbers do not
  establish operation on a physical 4GiB GPU or pixel equality.
- Nested `reset_peak_memory_stats` invalidates an enclosing peak measurement.
  This migration resets allocator peaks once, before loading inputs/model.

## C++ implementation

`src/sampler.cpp` ports `functions/denoising.py` and the FFT/deblur and kernel
update paths of `functions/svd_replacement.py`. Included upstream source is
pinned at commit `66d2a989af4135c4d8bfbff2d1054b32e890b437` under its MIT license.
The U-Net is executed by LibTorch from a TorchScript export of the original
FP32 model. No Python interpreter is used by the C++ inference process.
Python tools prepare artifacts and evaluate results; this is not a rewrite
of the U-Net module definitions in the C++ frontend.

- `at::globalContext().setBenchmarkCuDNN(false)` runs before model loading.
- No `y_0s`, `xs`, `x0_preds`, or kernel trajectory is retained. Only the current
  sample and kernel are needed for upstream's default `last=True` output.
- Langevin H-updates explicitly enable autograd inside the no-grad sampler.
  InferenceMode would prevent the needed gradients, so it is not used.
- The documented linear beta schedule and unconditional RGB FP32 models
  are supported by the preparation/export tools. FP16/AMP/INT8/PCA and
  classifier-guided models are outside this baseline.
- Exact observations, initial noise, and uncertain kernels are supplied as
  tensors. Dataset/motion-blur preprocessing is not silently reimplemented.
- `same`, `same_interp`, `full`, Adam/Langevin and resampling options are ported.
- CPU and GPU RNG sequences need not match; importing exact starting arrays
  is necessary but insufficient for pixel-identical stochastic GPU sampling.
  For strict stochastic parity use matching per-step random tensors in a
  separate validation experiment. Do not equate rounded PSNR with equality.
- LibTorch/PyTorch 2.6 CPU is the current validation toolchain; the user's
  reported reference used PyTorch 1.10.1/CUDA 11.3. Cross-version GPU results
  must be measured rather than presumed equal.

## Batch-1 shape defect

Upstream calculates FFT size as `nextpow2(img_dim + kernel.shape[0] - 1)`.
For kernels shaped `[B,64,64]`, `shape[0]` is B, not the kernel spatial size.
With N=256 and B=1 it chooses FFT=256. Cropping from offset 32 yields 224x224.
With B=2 it chooses FFT=512 and produces the intended 256x256 observation.
This explains the reported 150528-element tensor (3*224*224) at the source
level. The migrated code uses `nextpow2(N + K - 1)`; no resizing is applied.
The corrected Batch-1 shape is tested; its full-resolution restoration quality
and GPU memory have not been measured. Keep batch 2 for the comparison baseline.

## Validation boundaries

CPU tests compare 15 FFT operator outputs against pinned Python calculations,
then final image/kernel results for three sampler cases: same_interp/Langevin,
same/Adam without final resampling, and full/Langevin with repeated DDRM.
The fixture uses a small nonzero six-channel denoiser (testing learned-variance
channel slicing) and reduced H-update iterations. It validates the sampler
port, not FFHQ checkpoint quality. The reference harness adapts only unused
optional imports and a hardcoded CUDA device to permit CPU execution.
The batch-1 shape regression and malformed-input rejection are separate tests.

The actual FFHQ/AFHQ checkpoint, GPU allocator statistics, process VRAM,
full-resolution PSNR/SSIM and 4GiB operation still require a GPU and model
artifacts. The cuDNN/CUDA compilation branch cannot be executed on the CPU-only
onboarding machine. Export validation checks a reduced real upstream U-Net with nonzero random
weights at four timestep values, and compares C++ sampling through that exported
model against eager Python. The custom AttentionBlock gradient-checkpoint
wrapper is bypassed only while tracing (identical inference forward); original
vendor files are not changed. A scripted wrapper enforces the exported batch and
image size. CTest's three tests passed; the largest observed image absolute
error in these fixtures was approximately 1.91e-6. The preparation -> Python
reference -> C++ -> pixel comparison/PNG tool workflow was also exercised with
small random-weight inputs; those metrics are not pretrained-image PSNR.

## GPU experiment protocol

1. Retain batch 2, FP32, 100 steps, history off and benchmark off. Record GPU,
   driver, PyTorch/LibTorch/CUDA/cuDNN versions, initial device usage, seed,
   exact image order, arrays and checkpoint hash.
2. Build with a matching CUDA-enabled LibTorch wheel/toolkit (see README).
   Export the checkpoint outside the measured inference process.
3. Run each measurement in a fresh C++ process. The C++ allocator peak scope
   includes input/model load and sampling, excludes output serialization, and
   resets exactly once. No warm-up forward hides the initial peak.
4. Use `tools/measure_gpu.py` for device/process sampled VRAM and preserve
   its CSV and process log. WSL may not expose process memory; unavailable
   values stay null. Do not subtract device idle usage and call it process peak.
5. Repeat comparable runs, evaluate per-image MSE/PSNR and pixel differences
   against the Python benchmark-off/history-off baseline. SSIM should be
   measured with the SAME version/settings as the reference (not implemented
   by this tool). Report latency distributions, not a single-run speedup.
6. Physical 4GiB-device execution is the final capacity check. C++ alone
   does not guarantee lower tensor/cuDNN workspace memory than Python.
