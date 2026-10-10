"""Exercise the real upstream UNet export using small random FP32 weights.

No pretrained restoration quality is claimed. Zero-initialized residual/output
weights are made nonzero so timestep and data dependence are actually tested.
"""
import copy
import argparse
from pathlib import Path
import sys
import subprocess
import types
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from common import ROOT, load_config, save_bundle, settings_tensor
from parity import reference_module
from export_model import create_model, export_model

def main(build, folder):
    torch.set_num_threads(2)
    torch.manual_seed(82)
    config = copy.deepcopy(load_config(ROOT / 'configs/ffhq_256_deblur.yml'))
    config['data']['image_size'] = 32
    config['model'].update(image_size=32, num_channels=32, num_head_channels=8,
                           channel_mult='1,2', num_heads=1)
    model = create_model(**config['model']).cpu().eval()
    with torch.no_grad():
        for param in model.parameters():
            if param.ndim > 1 and param.abs().sum() == 0:
                param.normal_(0, 0.01)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint, exported = folder / 'random_unet.pt', folder / 'random_unet.ts'
    torch.save(model.state_dict(), checkpoint)
    export_model(config, checkpoint, exported, batch=2)
    loaded = torch.jit.load(str(exported))
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        for step in (0, 10, 500, 990):
            t = torch.full((2,), float(step))
            torch.testing.assert_close(loaded(x, t), model(x, t), atol=1e-5, rtol=1e-4)
        assert (loaded(x, torch.zeros(2)) - loaded(x, torch.full((2,), 990.0))).abs().max() > 1e-6
        assert loaded(x, torch.zeros(2)).abs().max() > 1e-6
    # Exercise the exported real UNet in the C++ sampler too, against the
    # original Python sampler using the unwrapped eager model.
    rh = reference_module('ref_h', ROOT / 'vendor/gibbsddrm/functions/svd_replacement.py',
                           ('from motionblur.motionblur import Kernel',))
    rd = reference_module('ref_d', ROOT / 'vendor/gibbsddrm/functions/denoising.py',
                           ('import torchvision.utils as tvu',), cpu=True)
    config['deblur'].update(timesteps=3, iter_Hupdate=1, iter_optim=1,
                           lr_Hupdate=1e-8, reg_H_gamma=0.1)
    kernel = torch.rand(2, 3, 3)
    kernel /= kernel.sum((-2, -1), keepdim=True)
    beta = torch.linspace(0.0001, 0.02, 12)
    h = rh.DeblurringArbitral2D(kernel.clone(), 3, 32, torch.device('cpu'), 'same_interp')
    y = h.H_fftconv(x, kernel)
    inputs, output = folder / 'inputs.ts', folder / 'result.ts'
    save_bundle(inputs, x=x, y=y, kernel=kernel, beta=beta, settings=settings_tensor(config))
    torch.manual_seed(1234)
    h = rh.DeblurringArbitral2D(kernel.clone(), 3, 32, torch.device('cpu'), 'same_interp')
    ns = types.SimpleNamespace(deblur=types.SimpleNamespace(**config['deblur']))
    expected = rd.sample_gibbsddrm(x, range(0, 12, 4), model, beta, h, y, 0.04,
                                  etaB=0.9, etaA=0.0, etaC=0.8, etaD=0.8, config=ns)[0][-1]
    subprocess.run([str(build / 'gibbsddrm'), '--model', str(exported), '--inputs', str(inputs),
                    '--output', str(output)], check=True)
    actual = torch.jit.load(str(output))
    torch.testing.assert_close(actual.image, expected, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(actual.kernel, h.kernel, atol=5e-5, rtol=5e-5)
    print(f'Real UNet C++ sampler vs eager Python: max image error={(actual.image-expected).abs().max().item():.3g}')
    try:
        loaded(torch.zeros(1, 3, 32, 32), torch.zeros(1))
    except (RuntimeError, torch.jit.Error):
        pass
    else:
        raise AssertionError('Exported shape constraint not enforced')
    print('Real upstream UNet random-weight export roundtrip passed at 4 timesteps; output and timestep dependence are nonzero')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'build')
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts/export')
    args = parser.parse_args()
    main(args.build.resolve(), args.artifacts.resolve())
