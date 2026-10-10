"""Run benchmark-off/history-off Python reference with the exact input bundle.

Use the same PyTorch build, device and seed as C++ for migration comparisons.
This does not recreate the original dataset loader or capture preprocessing RNG.
"""
import argparse
from pathlib import Path
import time
import types
import numpy as np
import torch
from common import ROOT, load_config, settings_tensor, linear_betas
from export_model import create_model

def load_reference():
    h_file = ROOT / 'vendor/gibbsddrm/functions/svd_replacement.py'
    d_file = ROOT / 'vendor/gibbsddrm/functions/denoising.py'
    # Keep vendored files unchanged. The removed imports are unused by the
    # supplied-tensor workflow. Adapt the hardcoded CUDA device; drop history.
    h_source = h_file.read_text().replace('from motionblur.motionblur import Kernel', '')
    d_source = d_file.read_text().replace('import torchvision.utils as tvu', '')
    d_source = d_source.replace(".to('cuda')", '.to(x.device)')
    d_source = d_source.replace('y_0s.append(y_0_interp)', '# history disabled')
    h_ns, d_ns = {'__name__': 'reference_blur'}, {'__name__': 'reference_denoising'}
    exec(compile(h_source, str(h_file), 'exec'), h_ns)
    exec(compile(d_source, str(d_file), 'exec'), d_ns)
    return h_ns['DeblurringArbitral2D'], d_ns['sample_gibbsddrm']

def run(config, checkpoint, input_path, output, device='cpu', seed=1234):
    torch.backends.cudnn.benchmark = False
    if config['model']['use_fp16'] or config['model']['class_cond']:
        raise ValueError('FP32 unconditional baseline required')
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)  # once, before load
    start = time.perf_counter()
    source = torch.jit.load(str(input_path), map_location='cpu')
    torch.testing.assert_close(source.settings, settings_tensor(config), rtol=0, atol=0)
    torch.testing.assert_close(source.beta, linear_betas(config), rtol=0, atol=0)
    model = create_model(**config['model']).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    x, y, kernel, beta = (getattr(source, name).to(device) for name in ('x', 'y', 'kernel', 'beta'))
    Blur, sample = load_reference()
    torch.manual_seed(seed)  # eager model initialization must not consume sampler RNG
    d = config['deblur']
    h = Blur(kernel, x.shape[1], x.shape[2], device, d['conv_type'])
    # This reference preserves the upstream FFT size bug; refuse input sizes
    # for which it differs from the corrected C++ operator.
    expected_fft = 1 << (x.shape[-1] + kernel.shape[-1] - 2).bit_length()
    if h.fft2_size != expected_fft:
        raise ValueError('Upstream FFT sizing bug triggered. Use baseline batch 2 / 256x256 / kernel64; do not compare this case.')
    ns = types.SimpleNamespace(deblur=types.SimpleNamespace(**d))
    seq = range(0, len(beta), len(beta) // d['timesteps'])
    for _ in range(d['iter_DDRM']):
        result = sample(x, seq, model, beta, h, y, 2 * d['sigma_0'],
                        etaB=d['etaB'], etaA=d['etaA'], etaC=d['etaC'], etaD=d['etaD'], config=ns)[0][-1]
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    print(f'load_and_sample_seconds={time.perf_counter()-start:.6f}')
    if device.type == 'cuda':
        print(f'peak_allocated_bytes={torch.cuda.max_memory_allocated(device)}')
        print(f'peak_reserved_bytes={torch.cuda.max_memory_reserved(device)}')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, result.cpu().numpy())
    np.save(output.with_name(output.stem + '_kernel.npy'), h.kernel.detach().cpu().numpy())
    print(f'Saved {output}; benchmark=False, history=False, FP32')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT / 'configs/ffhq_256_deblur.yml'))
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--inputs', required=True)
    p.add_argument('--output', required=True, help='Reference final restoration .npy')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=1234)
    a = p.parse_args()
    torch.set_num_threads(2)
    run(load_config(a.config), a.checkpoint, a.inputs, a.output, a.device, a.seed)
