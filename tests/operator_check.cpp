#include "sampler.h"
#include <iostream>
int main(int argc, char** argv) {
  try {
    TORCH_CHECK(argc == 2, "Expected reference fixture path");
    configure_backend();
    auto f = torch::jit::load(argv[1]);
    auto x = f.attr("x").toTensor(), k = f.attr("kernel").toTensor();
    for (auto mode : {"same", "full", "same_interp"}) {
      torch::manual_seed(91);
      Blur h(k.clone(), x.size(1), x.size(2), mode);
      std::vector<std::pair<std::string, Tensor>> outputs = {
          {"vt", h.Vt(x)}, {"vvt", h.V(h.Vt(x))},
          {"u", h.U(h.Vt(x) * h.singulars())}, {"conv", h.fftconv(x, k)}};
      auto y = f.attr(std::string(mode) + "_y").toTensor();
      outputs.emplace_back("ut", h.Ut(y));
      for (const auto& item : outputs) {
        auto expected = f.attr(std::string(mode) + "_" + item.first).toTensor();
        TORCH_CHECK(torch::allclose(item.second, expected, 2e-5, 2e-6), mode, " ", item.first, " mismatch; max error ", (item.second - expected).abs().max());
      }
      std::cout << mode << ": 5 operator comparisons passed\n";
    }
    // Regression: original kernel.shape[0] gives FFT=256, output=224.
    auto one = torch::ones({1, 64, 64}) / (64 * 64);
    Blur h(one, 3, 256, "same_interp");
    TORCH_CHECK(h.fft_size == 512, "Batch-1 FFT regression");
    auto y = h.fftconv(torch::zeros({1, 3, 256, 256}), one);
    TORCH_CHECK(y.sizes() == torch::IntArrayRef({1, 3, 256, 256}), "Batch-1 crop regression");
    std::cout << "batch=1 kernel=64 image=256: FFT=512, output=256x256\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
