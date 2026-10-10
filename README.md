# GibbsDDRM C++ migration

A LibTorch C++ implementation of Sony GibbsDDRM's FFT blur operator, DDRM
sampler, and Langevin/Adam kernel updates. The original FP32 U-Net is exported
to TorchScript and executed by the C++ inference process.

The baseline uses **cuDNN benchmark off, history off, batch 2, and FP32**.

Upstream: [sony/gibbsddrm](https://github.com/sony/gibbsddrm), commit
`66d2a989af4135c4d8bfbff2d1054b32e890b437`. Selected source files and the MIT
license are preserved in `vendor/gibbsddrm/`.

## Documentation

- [GPU memory analysis](docs/gpu-memory/README.md): measured memory changes and
  the cuDNN convolution selection path when benchmark is disabled.
- [Implementation and CPU validation](docs/MIGRATION.md): ported calculations,
  the batch-1 FFT sizing correction, and numerical comparisons.

## Build and test

The validated toolchain is Python 3.12, PyTorch/LibTorch 2.6.0 CPU, GCC 14.2,
CMake 3.31.10, and Ninja 1.13.0.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tools.txt cmake==3.31.10 ninja==1.13.0
cmake -S . -B build -G Ninja \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DPython3_EXECUTABLE="$(command -v python)" -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 1
ctest --test-dir build --output-on-failure
```

In the prepared Codex environment, use
`export PATH=/workspace/.cloud-tools/cpp/bin:$PATH` instead of creating a venv.
The three CTest checks passed. Small CPU fixtures produced a maximum absolute
image difference of approximately `1.91e-6` against the original Python calculations.

## Prepare inputs and run

Provide the dataset's original checkpoint (`ffhq_10m.pt` for FFHQ or
`afhqdog_p2.pt` for AFHQ) and an NPZ file containing these FP32 arrays:

| Array | Shape | Meaning |
| --- | --- | --- |
| `x` | `[B,3,256,256]` | Initial noise |
| `y` | `[B,3,256,256]` for `same_interp` | Observed image after the original preprocessing and corruption |
| `kernel` | `[B,K,K]` | Initial uncertain blur kernel, before H-updates |
| `clean` (optional) | `[B,3,256,256]` | Clean reference image in `[-1,1]`, for PSNR |

Save the arrays at the original `sample_image` entry point:

```python
import numpy as np
np.savez('inputs.npz',
         x=x.detach().cpu().numpy(),
         y=y_0.detach().cpu().numpy(),
         kernel=H_funcs.kernel.detach().cpu().numpy())
```

For FFHQ:

```bash
python tools/export_model.py --config configs/ffhq_256_deblur.yml \
  --checkpoint /path/to/ffhq_10m.pt --output artifacts/ffhq.ts --batch 2
python tools/prepare_inputs.py --config configs/ffhq_256_deblur.yml \
  --arrays /path/to/inputs.npz --output artifacts/inputs.ts
./build/gibbsddrm --model artifacts/ffhq.ts --inputs artifacts/inputs.ts \
  --output artifacts/result.ts --device cpu --seed 1234
python tools/evaluate.py --result artifacts/result.ts --inputs artifacts/inputs.ts \
  --png-dir artifacts/images
```

The export enforces its batch and image size and checks multiple timestep
values against the eager model. Input preparation preserves the supplied
observations and rejects shape mismatches without resizing.
