# GibbsDDRM C++ migration

Sony GibbsDDRM의 FFT 블러 연산, DDRM 샘플링, Langevin/Adam 커널 갱신을
LibTorch C++로 이식했습니다. U-Net은 원본 FP32 체크포인트를 TorchScript로
변환하여 C++에서 실행합니다. 추론 프로세스에는 Python이 필요하지 않습니다.

기준은 **cuDNN benchmark OFF + history OFF + Batch 2 + FP32**입니다.
FP16/INT8/PCA 등의 근사 최적화는 포함하지 않았습니다. C++ 전환 자체가
4GiB VRAM 실행을 보장하지는 않습니다.

원본: [sony/gibbsddrm](https://github.com/sony/gibbsddrm), commit
`66d2a989af4135c4d8bfbff2d1054b32e890b437`.
출처·라이선스는 `vendor/gibbsddrm/`에 있습니다.
사용자 Word 보고서의 분석과 검증 범위는 [docs/MIGRATION.md](docs/MIGRATION.md)에 정리했습니다.

## 현재 클라우드에서 빌드·테스트

Python 3.12 / PyTorch 2.6.0 CPU / GCC 14.2 / CMake 3.31.10 / Ninja 1.13.0으로
검증했습니다. 현재 머신에는 GPU/CUDA가 없습니다.

```bash
cd /workspace/GibbsDDRM-migration
export PATH=/workspace/.cloud-tools/cpp/bin:$PATH
cmake -S . -B build -G Ninja \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DPython3_EXECUTABLE="$(command -v python)" -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 1
ctest --test-dir build --output-on-failure
```

CTest 3개가 통과했습니다. 내부적으로 FFT 출력 15개, 샘플러 3개 경로,
실제 U-Net 구조의 임의 가중치 TorchScript 변환 및 C++ 샘플러 통합,
Batch 1 FFT 크기 회귀, 잘못된 관측 크기 거부를 검증합니다.
작은 임의 입력의 최대 절대 영상 오차는 약 `1.91e-6`였습니다.
이는 **사전학습 FFHQ 복원 품질 또는 GPU 피크 검증 결과가 아닙니다**.

다른 머신에서 CPU 의존성을 준비할 경우 Python 3.10 이상을 사용합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tools.txt cmake==3.31.10 ninja==1.13.0
# 위 CMake 명령으로 빌드·검증합니다.
```

## 실제 FFHQ/AFHQ 추론

실제 체크포인트와 관측 배열은 이 저장소에 포함하지 않았습니다.
FFHQ 기준 `ffhq_10m.pt`가 필요하며 원본 README의 모델 배포 경로를 사용합니다.
FFHQ 체크포인트로 AFHQ 모델을 대신 실행하지 않습니다.

먼저 원본 Python의 `sample_gibbsddrm` 호출 직전, 또는 `sample_image` 진입 시점에
아래 텐서를 NPZ로 저장합니다. 관측 모델과 데이터 전처리를 보존하려는 절차입니다.
원본을 영구 수정할 필요 없이 디버거나 별도 실행 래퍼에서도 저장할 수 있습니다.

```python
# x: 초기 노이즈 [B,3,256,256]
# y_0: 실제 노이즈 관측 [B,3,256,256] (same_interp)
# H_funcs: 초기 불확실 블러 연산자 (Hupdate 전)
import numpy as np
np.savez('inputs.npz',
         x=x.detach().cpu().numpy(),
         y=y_0.detach().cpu().numpy(),
         kernel=H_funcs.kernel.detach().cpu().numpy())
# 선택: 동일 전처리의 clean 원본 영상 [-1,1]도 저장하면 PSNR을 계산할 수 있습니다.
```

256×256 관측 영상이 224×224로 들어오면 자동 리사이즈하지 않고 거부합니다.
원본 Batch 1 오류의 원인은 FFT 크기 계산에 배치 크기를 사용한 것입니다.
C++에서는 커널 공간 크기를 사용하도록 수정했습니다. 비교 기준은 Batch 2로 유지합니다.

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

변환 시 Batch·영상 크기는 고정하며, 여러 timestep에서 eager/변환 결과를 비교합니다.
U-Net의 custom gradient checkpoint는 내보내기 시에만 동일 forward의 직접 호출로
대체합니다. 원본 파일은 그대로 보존됩니다. C++ kernel Hupdate에는 autograd가 유지됩니다.
CPU 전체 256×256 실행은 느릴 수 있습니다. GPU 실행은 아래 절차를 사용합니다.

## GPU 빌드와 메모리 비교

GPU 머신의 드라이버와 호환되는 CUDA-enabled PyTorch/LibTorch 및 CUDA toolkit이 필요합니다.
다음은 PyTorch 2.6 / CUDA 12.4 조합의 예시이며, 현재 CPU 머신에서 검증하지 않았습니다.
원래 PyTorch 1.10.1/CUDA 11.3 환경은 별도로 보존합니다.

```bash
python3 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-tools.txt cmake==3.31.10 ninja==1.13.0
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DPython3_EXECUTABLE="$(command -v python)" -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --parallel 1
ctest --test-dir build-cuda --output-on-failure
python tools/measure_gpu.py --gpu 0 --interval 0.1 \
  --csv artifacts/cpp_memory.csv --log artifacts/cpp_run.log -- \
  ./build-cuda/gibbsddrm --model artifacts/ffhq.ts --inputs artifacts/inputs.ts \
  --output artifacts/result.ts --device cuda:0 --seed 1234
```

**동일 GPU·PyTorch 버전·입력·seed**로
비교합니다. 아래 Python 기준 도구는 benchmark와 `y_0s` history를 모두 끄고,
저장된 입력부터 다시 실행합니다. 데이터 생성 RNG의 영향을 비교에서 분리합니다.

```bash
python tools/measure_gpu.py --gpu 0 --interval 0.1 \
  --csv artifacts/python_memory.csv --log artifacts/python_run.log -- \
  python tools/python_baseline.py --config configs/ffhq_256_deblur.yml \
  --checkpoint /path/to/ffhq_10m.pt --inputs artifacts/inputs.ts \
  --output artifacts/python_result.npy --device cuda:0 --seed 1234
python tools/evaluate.py --result artifacts/result.ts --inputs artifacts/inputs.ts \
  --reference artifacts/python_result.npy
```

C++ 로그의 `peak_allocated_bytes`와 `peak_reserved_bytes`는 **입력·모델 로딩부터
샘플링 종료까지**, 한 번만 초기화한 allocator 피크입니다. 결과 직렬화는 범위 밖입니다.
CSV의 장치 전체/해당 프로세스 MiB와 합산하지 않습니다. WSL에서 프로세스별 사용량이
제공되지 않으면 빈 값으로 기록합니다. 100ms 표본도 짧은 피크를 놓칠 수 있습니다.
`--gpu`는 nvidia-smi 물리 장치 번호이고 `--device`는 CUDA의 논리 장치 번호입니다.
`CUDA_VISIBLE_DEVICES`를 사용한다면 둘을 확인합니다.

반복 실행의 이미지별 MSE·PSNR·픽셀 차이와 지연시간·메모리를 함께 비교합니다.
SSIM은 원본과 같은 버전·설정으로 별도 계산해야 합니다. 같은 seed라도 CPU와 GPU,
PyTorch 버전 간의 RNG·수치 결과가 일치한다고 가정하지 않습니다.
**실제 체크포인트의 PSNR, GPU 경로 및 물리 4GiB 실행 가능성은 아직 미검증입니다.**
