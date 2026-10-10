#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <string>
using torch::Tensor;

struct Settings {
  int steps = 100, iterations = 1, h_updates = 3, optim_steps = 500;
  double sigma = 0.04, eta_a = 0, eta_b = 0.9, eta_c = 0.8, eta_d = 0.8;
  double h_start = 0.7, lr = 1e-11, gamma = 1000;
  bool resample = true;
  std::string conv = "same_interp", algorithm = "langevin", regularizer = "norm";
};

class Blur {
 public:
  Tensor kernel, initial;
  int64_t image_size, channels, fft_size, kh, kw;
  std::string conv;
  Blur(Tensor k, int64_t c, int64_t size, std::string mode);
  void update_kernel(Tensor k);
  Tensor V(Tensor v) const;
  Tensor Vt(Tensor v) const;
  Tensor U(Tensor v) const;
  Tensor Ut(Tensor v) const;
  Tensor singulars() const;
  Tensor fftconv(Tensor x, Tensor k) const;
  Tensor interpolate(Tensor y, Tensor x, double sigma) const;
  void update_H(Tensor y, Tensor x, const Settings& s);
 private:
  Tensor phase_, singular_, perm_;
  Tensor permute(Tensor x, bool inverse = false) const;
};

Tensor compute_alpha(Tensor beta, Tensor t);
struct Vars { Tensor uy, singular, sigma, inverse_y, init; };
Vars calc_vars(Blur& h, Tensor x, int last, Tensor beta, Tensor y, double sigma,
               Tensor x0 = {});
Tensor update_x(Blur& h, const Vars& v, Tensor x, Tensor x0, Tensor epsilon,
                Tensor alpha, Tensor next_alpha, const Settings& s);
Tensor sample(Tensor x, Tensor beta, Tensor y, Blur& h, torch::jit::Module& model,
              const Settings& s);
void configure_backend();
