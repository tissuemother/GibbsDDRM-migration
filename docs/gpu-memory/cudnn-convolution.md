# Which convolution algorithm does benchmark=False select?

`benchmark=False` switches cuDNN algorithm selection from a timed search to a
heuristic selection path. It does not specify one fixed convolution algorithm
for the entire U-Net. The supplied experiment report has no per-layer algorithm
IDs or execution-plan identifiers, so it does not identify the exact kernels
used in those runs.

## PyTorch 1.10.1: the experiment's version

The regular cuDNN v7 API forward path in PyTorch 1.10.1 behaves as follows on a
cache miss, when the deterministic-only shortcut is not active:

| Setting | API | Selection |
| --- | --- | --- |
| `benchmark=True` | `cudnnFindConvolutionForwardAlgorithmEx` | Times candidate implementations using the input/output tensors and search workspace |
| `benchmark=False` | `cudnnGetConvolutionForwardAlgorithm_v7` | Obtains cuDNN's heuristic candidate list without the timed find operation |

PyTorch filters candidates with successful cuDNN status, applies the requested
determinism constraint, and tries candidates in the returned order. For each
candidate it allocates that candidate's workspace and calls
`cudnnConvolutionForward` with `fwdAlgPerf.algo` and `fwdAlgPerf.mathType`.
The first candidate that executes successfully is cached for matching
convolution parameters. An allocation or cuDNN failure can cause the next
candidate to be tried. A valid cached choice can be reused before selection.

Consequently, the actual choice is made per convolution configuration. Different
U-Net layers can use different algorithms. The memory reduction alone cannot
establish that all layers switched to GEMM, Winograd, or FFT.

### Why the default constant is not the answer

The same source defines:

```cpp
DEFAULT_ALGO = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
bool only_use_default = args.params.deterministic && !benchmark;
```

This fixed-default path requires both `deterministic=True` and
`benchmark=False`. Disabling benchmark alone does not establish use of
`IMPLICIT_PRECOMP_GEMM`. The migration changes the benchmark flag and does not
set the deterministic flag.

### cuDNN API version versus library version

A cuDNN 8 library can execute PyTorch's cuDNN v7 API path. PyTorch 1.10.1 gates
its separate experimental v8 forward path behind the build-time
`USE_EXPERIMENTAL_CUDNN_V8_API` definition and cuDNN version >= 8. That
experimental path obtains INSTANT heuristic and fallback engine configurations
and caches a configuration that executes successfully. The report's CUDA 11.3
version alone does not identify the cuDNN API path or algorithm ID.

## PyTorch 2.6: the migration's source-level selection behavior

The migration calls:

```cpp
at::globalContext().setBenchmarkCuDNN(false);
```

It does not pass a convolution algorithm enum to cuDNN. In PyTorch 2.6's cuDNN
v8 implementation, the non-benchmark path first reuses a cached execution plan
when available. Otherwise it obtains heuristic engine configurations, attempts
to build and execute a valid plan, and uses fallback configurations if needed.
The benchmark path instead generates and times execution plans.

The two versions share the distinction between heuristic selection and timed
search. A v7 algorithm enum and a v8 engine/plan identifier are different
representations; a name from one path cannot be inferred for the other.

## Scope in GibbsDDRM

This selection concerns the U-Net's cuDNN convolutions. GibbsDDRM's blur
operator uses `torch.fft.fft2`/`ifft2`; those FFT calculations are separate from
cuDNN's convolution algorithm selection. The migration preserves that FFT blur
calculation while disabling cuDNN benchmark before model loading.

## Source references

- [PyTorch v1.10.1 Conv_v7.cpp: selection, workspace, and execution](https://github.com/pytorch/pytorch/blob/v1.10.1/aten/src/ATen/native/cudnn/Conv_v7.cpp#L260)
- [PyTorch v1.10.1 AlgoIterator: cache and deterministic shortcut](https://github.com/pytorch/pytorch/blob/v1.10.1/aten/src/ATen/native/cudnn/Conv_v7.cpp#L475)
- [PyTorch v1.10.1 Macros.h: experimental v8 build gate](https://github.com/pytorch/pytorch/blob/v1.10.1/aten/src/ATen/native/cudnn/Macros.h)
- [PyTorch v1.10.1 Conv_v8.cpp: heuristic/fallback configurations](https://github.com/pytorch/pytorch/blob/v1.10.1/aten/src/ATen/native/cudnn/Conv_v8.cpp#L148)
- [PyTorch v2.6.0 Conv_v8.cpp: non-benchmark execution-plan selection](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1008)
