# Implementation and CPU validation

## Source and execution path

The port follows Sony GibbsDDRM commit
`66d2a989af4135c4d8bfbff2d1054b32e890b437`. The selected upstream Python files
are preserved unchanged with their MIT license in `vendor/gibbsddrm/`.

`src/sampler.cpp` implements the FFT blur operator, alpha calculation, DDRM
updates, and Langevin/Adam kernel updates from `functions/denoising.py` and
`functions/svd_replacement.py`. LibTorch executes the original U-Net from a
TorchScript export. The inference executable runs entirely in C++.

`configure_backend()` sets `at::globalContext().setBenchmarkCuDNN(false)` before
model loading. The sampler retains the current sample and kernel, without
`y_0s`, `xs`, `x0_preds`, or kernel trajectories. Kernel updates enable autograd
inside the no-grad sampling loop. The FP32 calculation path is retained.

For export, the upstream AttentionBlock's custom gradient-checkpoint wrapper
is temporarily replaced by a direct call to the same inference forward
function. The original source files remain unchanged. A scripted wrapper
checks the exported batch and image dimensions.

## Batch-1 FFT sizing correction

Upstream computes the FFT size from
`nextpow2(img_dim + kernel.shape[0] - 1)`. For a kernel shaped `[B,64,64]`,
`kernel.shape[0]` is the batch size.

With a 256x256 image and batch 1, the resulting FFT size is 256. Cropping from
kernel-center offset 32 then produces 224x224, explaining the reported
150528-element tensor (`3 * 224 * 224`). Batch 2 produces FFT size 512 and a
256x256 observation.

The C++ implementation uses the spatial kernel size:
`nextpow2(img_dim + kernel_size - 1)`. The regression test confirms FFT size
512 and a 256x256 output for batch 1 with a 64x64 kernel. Input validation
rejects a mismatched observation without padding or resizing.

## Executed checks

The CPU toolchain was GCC 14.2, PyTorch/LibTorch 2.6.0, Python 3.12,
CMake 3.31.10, and Ninja 1.13.0. All three CTest checks passed:

| Check | Verified behavior |
| --- | --- |
| `sampler_parity` | 15 FFT operator comparisons; same_interp/Langevin, same/Adam without final resampling, and full/Langevin with repeated DDRM; batch-1 shape regression; malformed-observation rejection |
| `unet_export` | Reduced original U-Net with nonzero random weights; export comparison at four timesteps; C++ sampling comparison against eager Python; fixed-shape enforcement |
| `benchmark_disabled` | cuDNN benchmark remains disabled |

The reference fixture adapts the hardcoded CUDA device and removes unused
optional imports to execute the original calculations on CPU. The largest
observed absolute image difference was approximately `1.91e-6`.

The input preparation, Python reference run, C++ run, pixel comparison, and PNG
output tools were also exercised with small random-weight inputs. GPU experiment
measurements from the supplied report are documented separately in
[gpu-memory/](gpu-memory/README.md).
