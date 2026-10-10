# Measured GPU memory changes

Source: `GibbsDDRM_GPUv0.2_2026-10-10.docx`, dated October 10, 2026.
These values describe the Python experiments in that report.

## Experiment configuration

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060, 8 GiB VRAM |
| Runtime | WSL Ubuntu, Python 3.9, PyTorch 1.10.1, CUDA 11.3 |
| Task | FFHQ blind deblurring, 256x256, two evaluation images |
| Effective batch | 2 |
| Precision | FP32 (`model.use_fp16=False`) |
| Sampling | 100 steps, `same_interp` |
| Kernel update | Langevin, start ratio 0.7, `iter_Hupdate=3` |

## A. Remove the unused interpolation history

In the `same_interp` path, `calc_vars_for_xupdate()` computes `U_t_y` from
`y_0_interp`, then appends that CUDA tensor to `y_0s`. Earlier history entries
are not read during sampling. Removing the append releases each interpolated
observation after its current use.

```python
U_t_y = H_funcs.Ut(y_0_interp)
# Removed: y_0s.append(y_0_interp)
```

| Measurement | History ON | History OFF | Change |
| --- | ---: | ---: | ---: |
| Allocated memory at sampling completion | 2,264.74 MiB | 395.68 MiB | -1,869.06 MiB (-82.5%) |
| Sampled device-wide peak | 7,803 MiB | 7,798 MiB | -5 MiB |
| Sampling time, single run | 185.47 s | 178.67 s | -6.80 s |
| Reported mean PSNR | 27.50 dB | approximately 27.51 dB | +0.01 dB at displayed precision |

The reported history contains 311 tensors of shape `[2,3,512,512]` in FP32.
Each is 6 MiB, so the retained tensors account for approximately 1,866 MiB.
This closely matches the 1,869.06 MiB reduction at sampling completion.
The first-forward peak remains approximately 4,288 MiB after history removal.

## B. Disable cuDNN benchmark, keeping history OFF

The comparison retains batch 2, FP32, 100 steps, and history OFF. The changed
setting is:

```python
torch.backends.cudnn.benchmark = False
```

| Measurement | Benchmark ON | Benchmark OFF | Change |
| --- | ---: | ---: | ---: |
| First-forward peak allocated | 4,288.12 MiB | 957.62 MiB | -3,330.50 MiB (-77.7%) |
| First-forward peak reserved | 5,162 MiB | 1,610 MiB | -3,552 MiB (-68.8%) |
| Sampled device-wide peak | 7,798 MiB | 4,792 MiB | -3,006 MiB (-38.5%) |
| First-forward profiler self CPU total | 14.165 s | 12.914 s | -1.251 s |
| Total time with profiling, single run | 203.35 s | 197.44 s | -5.91 s |
| Reported mean PSNR | 27.51 dB | 27.51 dB | Same displayed value |

The first-forward reduction is consistent with eliminating the timed cuDNN
algorithm search and its temporary workspace. The PyTorch 1.10.1 source
explicitly notes that benchmarking can use several GB of memory. The selected
convolution still executes normally and may require its own workspace; disabling
benchmark removes the timed search rather than all convolution workspace.

The two changes address different lifetimes: history removal eliminates
retained tensors across sampling steps, while benchmark OFF reduces the
transient first-forward peak.

## Reading the measurements

Allocated memory describes active PyTorch allocations; reserved memory describes
allocator-held memory; `nvidia-smi` reports device-wide usage. These values are
separate and must not be added together.

The device measurements were sampled at approximately 500 ms intervals. Initial
device usage was 827 MiB for benchmark ON and 1,120 MiB for OFF. Subtracting
these initial values from device peaks does not produce a measured per-process
peak. The rounded PSNR values describe aggregate quality, not pixel equality.
Reported timing differences are single-run observations.

The report also identifies nested `reset_peak_memory_stats()` calls as a source
of understated enclosing peaks. The C++ measurement code uses one reset before
input/model loading and records allocator peaks through sampling completion.

See [cuDNN convolution selection](cudnn-convolution.md) for the source-level
explanation of the benchmark setting.
