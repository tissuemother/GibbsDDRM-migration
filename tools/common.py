from pathlib import Path
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

class Bundle(torch.nn.Module):
    def __init__(self, **tensors):
        super().__init__()
        for name, tensor in tensors.items():
            self.register_buffer(name, tensor.detach().cpu().contiguous())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

def save_bundle(path, **tensors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(Bundle(**tensors)).save(str(path))

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def settings_tensor(config):
    d = config['deblur']
    return torch.tensor([
        d['timesteps'], d['iter_DDRM'], d['iter_Hupdate'], d['iter_optim'], 2 * d['sigma_0'],
        d['etaA'], d['etaB'], d['etaC'], d['etaD'], d['Hupdate_start'], float(d['lr_Hupdate']),
        d['reg_H_gamma'], d['resample_after_Hupdate'],
        {'same_interp': 0, 'same': 1, 'full': 2}[d['conv_type']],
        {'langevin': 0, 'optim': 1}[d['alg_Hupdate']], {'norm': 0, 'diff_norm': 1}[d['reg_H_type']],
    ], dtype=torch.float64)

def linear_betas(config):
    # Match upstream numpy float64 linspace, then float32 conversion.
    import numpy as np
    d = config['diffusion']
    if d['beta_schedule'] != 'linear':
        raise ValueError('Only the documented linear schedule is supported by this preparation tool')
    return torch.from_numpy(np.linspace(d['beta_start'], d['beta_end'], d['num_diffusion_timesteps'])).float()
