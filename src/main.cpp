#include "sampler.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <chrono>
#include <map>
#include <cmath>
#include <limits>
#include <iostream>
#ifdef GIBBSDDRM_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#endif

static void validate_inputs(const Tensor& x, const Tensor& y, const Tensor& k, const Tensor& beta,
                            const Settings& s) {
  TORCH_CHECK(x.dim() == 4 && x.size(1) == 3 && x.size(2) == x.size(3), "x must be [B,3,N,N]");
  TORCH_CHECK(k.dim() == 3 && k.size(0) == x.size(0) && k.size(1) == k.size(2), "kernel must be [B,K,K]");
  auto n = x.size(2), out = s.conv == "full" ? n + k.size(1) - 1 : n;
  TORCH_CHECK(y.dim() == 4 && y.size(0) == x.size(0) && y.size(1) == 3 && y.size(2) == out && y.size(3) == out,
              "Observation shape mismatch: expected ", out, "x", out, "; no implicit resizing/padding");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 && y.scalar_type() == torch::kFloat32 && k.scalar_type() == torch::kFloat32,
              "Baseline requires FP32 inputs");
  TORCH_CHECK(beta.dim() == 1 && beta.scalar_type() == torch::kFloat32 && (beta > 0).all().item<bool>() && (beta < 1).all().item<bool>(), "Invalid betas");
  TORCH_CHECK(torch::isfinite(x).all().item<bool>() && torch::isfinite(y).all().item<bool>() && torch::isfinite(k).all().item<bool>(), "Nonfinite inputs");
  TORCH_CHECK(s.sigma > 0 && s.lr > 0 && s.iterations > 0 && s.h_updates >= 0 && s.optim_steps >= 0, "Invalid sampler settings");
  TORCH_CHECK(s.eta_a >= 0 && s.eta_c >= 0 && s.eta_a + s.eta_c <= 1 && s.eta_b >= 0 && s.eta_b <= 1 && s.eta_d >= 0 && s.eta_d <= 1,
              "Eta values may produce negative variances");
}

int main(int argc, char** argv) {
  try {
    configure_backend();
    if (argc == 2 && std::string(argv[1]) == "--check-backend") {
      std::cout << "cudnn.benchmark=false\n";
      return at::globalContext().benchmarkCuDNN() ? 1 : 0;
    }
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--help")) {
      std::cout << "gibbsddrm --model model.ts --inputs inputs.ts --output result.ts [--device cpu|cuda:0] [--seed 1234]\n";
      return argc == 1 ? 1 : 0;
    }
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
      TORCH_CHECK(i + 1 < argc, "Missing argument value");
      std::string key(argv[i]);
      TORCH_CHECK(key == "--model" || key == "--inputs" || key == "--output" || key == "--device" || key == "--seed", "Unknown option: ", key);
      TORCH_CHECK(!args.count(key), "Duplicate option: ", key);
      args[key] = argv[i + 1];
    }
    for (auto key : {"--model", "--inputs", "--output"}) TORCH_CHECK(args.count(key), "Missing ", key);
    torch::Device device(args.count("--device") ? args["--device"] : "cpu");
    if (device.is_cuda() && !device.has_index()) device = torch::Device(torch::kCUDA, 0);
    TORCH_CHECK(device.is_cpu() || device.is_cuda(), "Only CPU/CUDA devices are supported");
    if (device.is_cuda()) TORCH_CHECK(torch::cuda::is_available(), "CUDA unavailable in this LibTorch build/machine");
#ifdef GIBBSDDRM_CUDA
    std::unique_ptr<c10::cuda::CUDAGuard> guard;
    if (device.is_cuda()) {
      guard = std::make_unique<c10::cuda::CUDAGuard>(device);
      torch::cuda::synchronize(device.index());
      // Exactly one reset before input/model load; no nested or warm-up reset.
      c10::cuda::CUDACachingAllocator::resetPeakStats(device.index());
    }
#endif
    auto start = std::chrono::steady_clock::now();
    torch::manual_seed(args.count("--seed") ? std::stoll(args["--seed"]) : 1234);
    auto bundle = torch::jit::load(args["--inputs"], device);
    auto values = bundle.attr("settings").toTensor().cpu().to(torch::kDouble).contiguous();
    TORCH_CHECK(values.numel() == 16, "Invalid settings bundle");
    TORCH_CHECK(torch::isfinite(values).all().item<bool>(), "Nonfinite settings");
    auto p = values.data_ptr<double>();
    for (int index : {0, 1, 2, 3, 12, 13, 14, 15})
      TORCH_CHECK(p[index] >= 0 && p[index] <= std::numeric_limits<int>::max() && std::floor(p[index]) == p[index], "Invalid integer setting");
    TORCH_CHECK(p[12] <= 1 && p[9] >= 0 && p[9] <= 1 && p[11] >= 0, "Invalid update setting");
    Settings s;
    s.steps=p[0]; s.iterations=p[1]; s.h_updates=p[2]; s.optim_steps=p[3]; s.sigma=p[4];
    s.eta_a=p[5]; s.eta_b=p[6]; s.eta_c=p[7]; s.eta_d=p[8]; s.h_start=p[9]; s.lr=p[10]; s.gamma=p[11];
    s.resample=p[12] != 0;
    TORCH_CHECK(p[13] >= 0 && p[13] <= 2 && (p[14] == 0 || p[14] == 1) && (p[15] == 0 || p[15] == 1), "Invalid mode codes");
    s.conv = p[13] == 0 ? "same_interp" : p[13] == 1 ? "same" : "full";
    s.algorithm = p[14] == 0 ? "langevin" : "optim";
    s.regularizer = p[15] == 0 ? "norm" : "diff_norm";
    auto x = bundle.attr("x").toTensor(), y = bundle.attr("y").toTensor();
    auto k = bundle.attr("kernel").toTensor(), beta = bundle.attr("beta").toTensor();
    validate_inputs(x, y, k, beta, s);
    auto model = torch::jit::load(args["--model"], device);
    model.eval();
    for (const auto& parameter : model.parameters()) TORCH_CHECK(!parameter.is_floating_point() || parameter.scalar_type() == torch::kFloat32, "Model must use FP32");
    TORCH_CHECK(!at::globalContext().benchmarkCuDNN(), "cuDNN benchmark changed during loading");
    Blur h(k, x.size(1), x.size(2), s.conv);
    std::cout << "cudnn.benchmark=false history=false fp32=true batch=" << x.size(0) << " fft_size=" << h.fft_size << '\n';
    auto result = sample(x, beta, y, h, model, s);
    if (device.is_cuda()) torch::cuda::synchronize(device.index());
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "load_and_sample_seconds=" << elapsed << '\n';
#ifdef GIBBSDDRM_CUDA
    if (device.is_cuda()) {
      auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device.index());
      constexpr auto all = static_cast<size_t>(c10::CachingDeviceAllocator::StatType::AGGREGATE);
      std::cout << "peak_allocated_bytes=" << stats.allocated_bytes[all].peak << '\n'
                << "peak_reserved_bytes=" << stats.reserved_bytes[all].peak << '\n';
    }
#endif
    TORCH_CHECK(torch::isfinite(result).all().item<bool>() && torch::isfinite(h.kernel).all().item<bool>(), "Nonfinite sampler output");
    torch::jit::Module output("GibbsDDRMResult");
    output.register_buffer("image", result.cpu());
    output.register_buffer("kernel", h.kernel.detach().cpu());
    output.save(args["--output"]);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
