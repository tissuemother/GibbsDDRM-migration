// Port of sony/gibbsddrm, MIT, Copyright (c) 2023 Sony Group Corporation.
#include "sampler.h"
#include <ATen/Context.h>
#include <cmath>
#include <iostream>
using namespace torch::indexing;

void configure_backend() {
  // Set before any model load/forward; do not enable deterministic algorithms
  // or change TF32/precision as part of this migration.
  at::globalContext().setBenchmarkCuDNN(false);
  TORCH_CHECK(!at::globalContext().benchmarkCuDNN(), "cuDNN benchmark must be off");
}

Blur::Blur(Tensor k, int64_t c, int64_t size, std::string mode)
    : kernel(k), initial(k.detach().clone()), image_size(size), channels(c),
      kh(k.size(-2)), kw(k.size(-1)), conv(std::move(mode)) {
  TORCH_CHECK(k.dim() == 3 && kh == kw, "Expected batch of square kernels [B,K,K]");
  TORCH_CHECK(conv == "same" || conv == "same_interp" || conv == "full", "Invalid convolution mode");
  fft_size = 1;
  // Upstream mistakenly uses batch size (kernel.shape[0]) here.
  while (fft_size < image_size + kh - 1) fft_size *= 2;
  update_kernel(k);
}

Tensor Blur::permute(Tensor x, bool inverse) const {
  auto idx = perm_;
  if (x.dim() == 3) idx = idx.unsqueeze(1).expand_as(x);
  if (inverse) return torch::zeros_like(x).scatter(-1, idx, x);
  return x.gather(-1, idx);
}

void Blur::update_kernel(Tensor k) {
  kernel = k;
  auto f = torch::fft::fft2(k, {fft_size, fft_size}, {-2, -1}, "ortho");
  auto perturb = f + 1e-10 * torch::randn_like(f);
  phase_ = (perturb / torch::abs(perturb)).reshape({k.size(0), -1});
  auto values = torch::abs(f * fft_size).reshape({k.size(0), -1});
  values.masked_fill_(values < 0.05, 0);
  auto sorted = values.sort(-1, true);
  singular_ = std::get<0>(sorted);
  perm_ = std::get<1>(sorted);
  phase_ = permute(phase_);
}

Tensor Blur::Vt(Tensor v) const {
  auto f = torch::fft::fft2(v.reshape({v.size(0), channels, image_size, image_size}),
                          {fft_size, fft_size}, {-2, -1}, "ortho");
  return (permute(f.reshape({v.size(0), channels, -1})) * phase_.unsqueeze(1)).reshape({v.size(0), -1});
}
Tensor Blur::V(Tensor v) const {
  auto f = permute(v.reshape({v.size(0), channels, -1}) / phase_.unsqueeze(1), true);
  auto image = torch::real(torch::fft::ifft2(f.reshape({v.size(0), channels, fft_size, fft_size}),
                                            c10::nullopt, {-2, -1}, "ortho"));
  return image.index({Slice(), Slice(), Slice(0, image_size), Slice(0, image_size)}).reshape({v.size(0), -1});
}
Tensor Blur::U(Tensor v) const {
  auto f = permute(v.reshape({v.size(0), channels, -1}), true);
  auto image = torch::real(torch::fft::ifft2(f.reshape({v.size(0), channels, fft_size, fft_size}),
                                            c10::nullopt, {-2, -1}, "ortho"));
  if (conv == "full") return image.index({Slice(), Slice(), Slice(0, image_size + kh - 1), Slice(0, image_size + kw - 1)});
  return image.index({Slice(), Slice(), Slice(kh / 2, kh / 2 + image_size), Slice(kw / 2, kw / 2 + image_size)});
}
Tensor Blur::Ut(Tensor v) const {
  if (conv != "same_interp") {
    auto image = torch::zeros({v.size(0), channels, fft_size, fft_size}, v.options());
    int64_t start_h = conv == "same" ? kh / 2 : 0;
    int64_t start_w = conv == "same" ? kw / 2 : 0;
    int64_t height = conv == "same" ? image_size : image_size + kh - 1;
    int64_t width = conv == "same" ? image_size : image_size + kw - 1;
    image.index_put_({Slice(), Slice(), Slice(start_h, start_h + height), Slice(start_w, start_w + width)},
                     v.reshape({v.size(0), channels, height, width}));
    v = image;
  }
  auto f = torch::fft::fft2(v, {fft_size, fft_size}, {-2, -1}, "ortho");
  return permute(f.reshape({v.size(0), channels, -1})).reshape({v.size(0), -1});
}
Tensor Blur::singulars() const { return singular_.repeat({1, channels}); }
Tensor Blur::fftconv(Tensor x, Tensor k) const {
  auto xf = torch::fft::fft2(x.reshape({x.size(0), channels, image_size, image_size}), {fft_size, fft_size}, {-2, -1}, "ortho");
  auto kf = torch::fft::fft2(k, {fft_size, fft_size}, {-2, -1}, "ortho").unsqueeze(1);
  auto image = torch::real(torch::fft::ifft2(kf * xf, c10::nullopt, {-2, -1}, "ortho")) * fft_size;
  if (conv == "full") return image.index({Slice(), Slice(), Slice(0, image_size + kh - 1), Slice(0, image_size + kw - 1)});
  return image.index({Slice(), Slice(), Slice(kh / 2, kh / 2 + image_size), Slice(kw / 2, kw / 2 + image_size)});
}
Tensor Blur::interpolate(Tensor y, Tensor x, double sigma) const {
  auto xf = torch::fft::fft2(x.reshape({x.size(0), channels, image_size, image_size}), {fft_size, fft_size}, {-2, -1}, "ortho");
  auto kf = torch::fft::fft2(kernel, {fft_size, fft_size}, {-2, -1}, "ortho").unsqueeze(1);
  auto image = torch::real(torch::fft::ifft2(kf * xf, c10::nullopt, {-2, -1}, "ortho")) * fft_size;
  image = image + sigma * torch::randn_like(image);
  image.index_put_({Slice(), Slice(), Slice(kh / 2, kh / 2 + image_size), Slice(kw / 2, kw / 2 + image_size)}, y);
  return image;
}

void Blur::update_H(Tensor y, Tensor x, const Settings& s) {
  torch::AutoGradMode enable(true); // Outer sampler uses NoGradGuard, not InferenceMode.
  kernel.requires_grad_(true);
  if (s.algorithm == "langevin") {
    for (int i = 0; i < s.optim_steps; ++i) {
      auto error = (y - fftconv(x, kernel)).norm();
      Tensor loss;
      if (s.regularizer == "norm") loss = error.square() / (2 * 0.04 * 0.04) + s.gamma * kernel.abs().sum();
      else loss = error.square() + s.gamma * (kernel - initial).norm();
      loss.backward();
      {
        torch::NoGradGuard no_grad;
        kernel.add_(kernel.grad(), -s.lr / 2);
        kernel.add_(torch::randn_like(kernel), std::sqrt(s.lr));
        kernel.mutable_grad().zero_();
      }
    }
  } else if (s.algorithm == "optim") {
    // Preserve the upstream per-batch Adam loop and optimizer resets.
    for (int64_t b = 0; b < x.size(0); ++b) {
      torch::optim::Adam optimizer({kernel}, torch::optim::AdamOptions(s.lr));
      for (int i = 0; i < s.optim_steps; ++i) {
        optimizer.zero_grad();
        auto kb = kernel.index({Slice(b, b + 1)});
        auto error = (y.index({b}) - fftconv(x.index({Slice(b, b + 1)}), kb)).norm();
        auto loss = error + s.gamma * (s.regularizer == "norm"
            ? kernel.index({b}).abs().sum() : (kernel.index({b}) - initial.index({b})).norm());
        loss.backward();
        optimizer.step();
      }
    }
  } else TORCH_CHECK(false, "Unknown H-update algorithm");
  kernel.requires_grad_(false);
  kernel = kernel / kernel.sum({-2, -1}, true);
  update_kernel(kernel);
}

Tensor compute_alpha(Tensor beta, Tensor t) {
  auto padded = torch::cat({torch::zeros({1}, beta.options()), beta});
  return (1 - padded).cumprod(0).index_select(0, t.to(torch::kLong) + 1).reshape({-1, 1, 1, 1});
}
Vars calc_vars(Blur& h, Tensor x, int last, Tensor beta, Tensor y, double sigma, Tensor x0) {
  auto uy = h.Ut(h.conv == "same_interp" ? h.interpolate(y, x0.defined() ? x0 : y, sigma) : y);
  // Deliberately no GPU history. Interpolated y is freed after the current update.
  auto singular = h.singulars();
  auto diag = torch::zeros({y.size(0), uy.size(1)}, x.options());
  diag.index_put_({Slice(), Slice(0, singular.size(1))}, singular);
  auto inv_y = uy / singular.index({Slice(), Slice(0, uy.size(1))});
  auto a = compute_alpha(beta, torch::full({x.size(0)}, last, x.options().dtype(torch::kLong)));
  auto noise = (1 - a).sqrt() / a.sqrt();
  auto large = singular * noise.reshape({x.size(0), 1}) > sigma;
  auto inv = torch::zeros_like(singular);
  inv.index_put_({large}, sigma / singular.index({large}));
  auto init = torch::zeros({x.size(0), singular.size(1)}, uy.options());
  init.index_put_({large}, uy.index({large}) / singular.index({large}));
  auto remaining = (noise.reshape({x.size(0), 1}).square() - inv.square()).clamp_min(0).sqrt();
  init = (init + remaining * h.Vt(x)) / noise.reshape({x.size(0), 1});
  return {uy, singular, diag, inv_y, init};
}

Tensor update_x(Blur& h, const Vars& v, Tensor x, Tensor x0, Tensor epsilon,
                Tensor /*alpha*/, Tensor next_alpha, const Settings& s) {
  auto sigma_next = ((1 - next_alpha).sqrt() / next_alpha.sqrt()).flatten()[0];
  auto vx0 = h.Vt(x0);
  auto svx0 = (vx0 * v.sigma).index({Slice(), Slice(0, v.uy.size(1))});
  auto before = (v.singular * sigma_next > s.sigma) & (v.singular > 1e-10);
  auto after = (v.singular * sigma_next < s.sigma) & (v.singular > 1e-10);
  auto ve = h.Vt(epsilon);
  auto out = vx0 + (sigma_next.square() - (sigma_next * s.eta_d).square()).sqrt() * ve
                 + sigma_next * s.eta_d * h.Vt(torch::randn_like(x));
  auto ua = ve * after;
  auto uc = ((v.uy - svx0) / s.sigma) * after;
  auto correlation = (ua * torch::conj(uc)).sum(-1).abs() / (ua.norm(2, {-1}) * uc.norm(2, {-1}) + 1e-10);
  auto stddev = sigma_next * (1 - s.eta_a * s.eta_a - s.eta_c * s.eta_c
                              - 2 * s.eta_a * s.eta_c * correlation).sqrt();
  auto noise_after = stddev.unsqueeze(1) * h.Vt(torch::randn_like(x));
  out.index_put_({after}, vx0.index({after}) + sigma_next * s.eta_a * ua.index({after})
                       + sigma_next * s.eta_c * uc.index({after}) + noise_after.index({after}));
  auto std_before = (sigma_next.square() - s.sigma * s.sigma / v.singular.index({before}).square() * s.eta_b * s.eta_b).sqrt();
  auto noise_before = h.Vt(torch::randn_like(x));
  out.index_put_({before}, v.inverse_y.index({before}) * s.eta_b + (1 - s.eta_b) * vx0.index({before})
                         + std_before * noise_before.index({before}));
  return (next_alpha.sqrt().flatten()[0] * h.V(out)).reshape_as(x);
}

static std::pair<Tensor, Tensor> estimate(torch::jit::Module& model, Tensor x, Tensor t, Tensor a) {
  auto e = model.forward({x, t}).toTensor();
  if (e.size(1) == 6) e = e.index({Slice(), Slice(0, 3)});
  TORCH_CHECK(e.sizes() == x.sizes(), "Denoiser output shape mismatch");
  return {(x - e * (1 - a).sqrt()) / a.sqrt(), e};
}
Tensor sample(Tensor x, Tensor beta, Tensor y, Blur& h, torch::jit::Module& model, const Settings& s) {
  torch::NoGradGuard no_grad;
  TORCH_CHECK(s.steps > 0 && s.steps <= beta.size(0), "Invalid sampling steps");
  int skip = beta.size(0) / s.steps;
  std::vector<int> seq;
  for (int i = 0; i < beta.size(0); i += skip) seq.push_back(i);
  // Python range.stop is the diffusion step count, even if skip does not divide it.
  auto max_steps = beta.size(0);
  auto initial_x = x;
  for (int iteration = 0; iteration < s.iterations; ++iteration) {
    auto vars = calc_vars(h, initial_x, seq.back(), beta, y, s.sigma);
    x = h.V(vars.init).reshape_as(initial_x);
    for (int pos = static_cast<int>(seq.size()) - 1; pos >= 0; --pos) {
      int i = seq[pos], j = pos ? seq[pos - 1] : -1;
      auto t = torch::full({x.size(0)}, i, x.options());
      auto nt = torch::full({x.size(0)}, j, x.options());
      auto a = compute_alpha(beta, t), na = compute_alpha(beta, nt);
      auto [x0, e] = estimate(model, x, t, a);
      vars = calc_vars(h, initial_x, seq.back(), beta, y, s.sigma, i < int(max_steps * 0.7) ? x0 : Tensor());
      auto next = update_x(h, vars, initial_x, x0, e, a, na, s);
      if (i <= max_steps * s.h_start && i != seq.front()) {
        for (int hu = 0; hu < s.h_updates; ++hu) {
          auto [next_x0, next_e] = estimate(model, next, nt, na);
          h.update_H(y, next_x0, s);
          if (hu == s.h_updates - 1 && !s.resample) continue;
          vars = calc_vars(h, initial_x, seq.back(), beta, y, s.sigma, x0);
          next = update_x(h, vars, initial_x, x0, e, a, na, s);
        }
      }
      x = next;
      // Keep only the current sample; no xs/x0_preds/kernels trajectory.
      std::cout << "iteration=" << iteration << " timestep=" << i << '\n';
    }
  }
  return x;
}
