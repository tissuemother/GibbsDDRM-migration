"""Numerical comparison against pinned Sony Python calculations, on CPU.

Only unused optional imports and the hardcoded `.to('cuda')` are adapted in the
reference harness. No mathematical formulas or assertions are changed. Batch 2
avoids the upstream FFT sizing bug for these small fixtures. A separate C++
regression checks the corrected Batch-1 256x256 / 64x64-kernel path.
"""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import types
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from common import ROOT, load_config, save_bundle, settings_tensor

class Denoiser(torch.nn.Module):
    def forward(self, x, t):
        epsilon = 0.1 * x + 0.0002 * t.reshape(-1, 1, 1, 1)
        return torch.cat((epsilon, torch.ones_like(epsilon)), dim=1)

def reference_module(name, file, removed=(), cpu=False):
    source = file.read_text()
    for statement in removed:
        source = source.replace(statement, '')
    if cpu:
        source = source.replace(".to('cuda')", '.to(x.device)')
    ns = {'__name__': name}
    exec(compile(source, str(file), 'exec'), ns)
    return types.SimpleNamespace(**ns)

def main(build, destination):
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    destination.mkdir(parents=True, exist_ok=True)
    ref_h = reference_module('ref_h', ROOT / 'vendor/gibbsddrm/functions/svd_replacement.py',
                             ('from motionblur.motionblur import Kernel',))
    ref_d = reference_module('ref_d', ROOT / 'vendor/gibbsddrm/functions/denoising.py',
                             ('import torchvision.utils as tvu',), cpu=True)
    model = torch.jit.script(Denoiser())
    model_path = destination / 'denoiser.ts'
    model.save(str(model_path))
    torch.manual_seed(3)
    x = torch.randn(2, 3, 8, 8)
    kernel = torch.rand(2, 3, 3)
    kernel /= kernel.sum((-2, -1), keepdim=True)
    fixture = {'x': x, 'kernel': kernel}
    for mode in ('same', 'full', 'same_interp'):
        torch.manual_seed(91)
        h = ref_h.DeblurringArbitral2D(kernel.clone(), 3, 8, torch.device('cpu'), mode)
        y = torch.randn(2, 3, 16, 16) if mode == 'same_interp' else h.H_fftconv(x, kernel)
        fixture.update({f'{mode}_y': y, f'{mode}_vt': h.Vt(x), f'{mode}_vvt': h.V(h.Vt(x)),
                        f'{mode}_u': h.U(h.Vt(x) * h.singulars()),
                        f'{mode}_ut': h.Ut(y), f'{mode}_conv': h.H_fftconv(x, kernel)})
    op_path = destination / 'operators.ts'
    save_bundle(op_path, **fixture)
    subprocess.run([str(build / 'operator_check'), str(op_path)], check=True)
    config = load_config(ROOT / 'configs/ffhq_256_deblur.yml')
    config['deblur'].update(timesteps=3, iter_Hupdate=2, iter_optim=2, lr_Hupdate=1e-7, reg_H_gamma=0.1)
    beta = torch.linspace(0.0001, 0.02, 12)
    cases = [
        ('same_interp', 'langevin', 'norm', True, 1),
        ('same', 'optim', 'diff_norm', False, 1),
        ('full', 'langevin', 'diff_norm', True, 2),
    ]
    for index, (mode, algorithm, regularizer, resample, iterations) in enumerate(cases):
        c = copy.deepcopy(config)
        c['deblur'].update(conv_type=mode, alg_Hupdate=algorithm, reg_H_type=regularizer,
                           resample_after_Hupdate=resample, iter_DDRM=iterations)
        torch.manual_seed(123)
        h = ref_h.DeblurringArbitral2D(kernel.clone(), 3, 8, torch.device('cpu'), mode)
        y = h.H_fftconv(x, kernel) + 0.04 * torch.randn_like(h.H_fftconv(x, kernel))
        input_path, output_path = destination / f'inputs_{index}.ts', destination / f'result_{index}.ts'
        save_bundle(input_path, x=x, y=y, kernel=kernel, beta=beta, settings=settings_tensor(c))
        # Both implementations seed immediately before constructing H.
        torch.manual_seed(1234)
        h = ref_h.DeblurringArbitral2D(kernel.clone(), 3, 8, torch.device('cpu'), mode)
        ns = types.SimpleNamespace(deblur=types.SimpleNamespace(**c['deblur']))
        expected = None
        for _ in range(iterations):
            expected = ref_d.sample_gibbsddrm(x, range(0, 12, 4), model, beta, h, y, 0.04,
                                            etaB=0.9, etaA=0.0, etaC=0.8, etaD=0.8, config=ns)[0][-1]
        subprocess.run([str(build / 'gibbsddrm'), '--model', str(model_path), '--inputs', str(input_path),
                        '--output', str(output_path), '--device', 'cpu', '--seed', '1234'], check=True)
        output = torch.jit.load(str(output_path))
        torch.testing.assert_close(output.image, expected, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(output.kernel, h.kernel, atol=3e-5, rtol=3e-5)
        print(f'{mode}/{algorithm}/{regularizer}/resample={resample}/iterations={iterations}: '
              f'image max_abs_error={(output.image - expected).abs().max().item():.3g}, '
              f'kernel max_abs_error={(output.kernel - h.kernel).abs().max().item():.3g}')
    # A malformed 224x224 observation must fail, not be silently resized.
    good = torch.jit.load(str(destination / 'inputs_0.ts'))
    bad_path = destination / 'invalid.ts'
    save_bundle(bad_path, x=good.x, y=torch.zeros(2, 3, 7, 7), kernel=good.kernel,
                beta=good.beta, settings=good.settings)
    failure = subprocess.run([str(build / 'gibbsddrm'), '--model', str(model_path), '--inputs', str(bad_path),
                              '--output', str(destination / 'invalid_result.ts')], capture_output=True, text=True)
    assert failure.returncode != 0 and 'Observation shape mismatch' in failure.stderr
    print('Malformed observation correctly rejected; no resizing/padding')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=ROOT / 'build')
    p.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts/parity')
    a = p.parse_args()
    main(a.build.resolve(), a.artifacts.resolve())
