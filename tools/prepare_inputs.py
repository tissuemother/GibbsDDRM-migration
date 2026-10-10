"""Package the EXACT observed y, initial noise x and initial uncertain kernel.

No synthetic observation, random motion kernel, padding or image resizing is
introduced. Export these arrays from the Python baseline to isolate migration
from dataset preprocessing and RNG differences between CPU and CUDA.
"""
import argparse
import numpy as np
import torch
from common import ROOT, load_config, linear_betas, save_bundle, settings_tensor

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT / 'configs/ffhq_256_deblur.yml'))
    p.add_argument('--arrays', required=True, help='NPZ with x, y, kernel; optional clean in [-1,1]')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    c = load_config(a.config)
    with np.load(a.arrays, allow_pickle=False) as arrays:
        tensors = {name: torch.from_numpy(arrays[name]).float() for name in ('x', 'y', 'kernel')}
        if 'clean' in arrays:
            tensors['clean'] = torch.from_numpy(arrays['clean']).float()
    x, y, k = (tensors[name] for name in ('x', 'y', 'kernel'))
    n = c['data']['image_size']
    if x.ndim != 4 or x.shape[1:] != (3, n, n):
        raise ValueError(f'x must be [B,3,{n},{n}]')
    if k.ndim != 3 or k.shape[0] != x.shape[0] or k.shape[1] != k.shape[2]:
        raise ValueError('kernel must be [B,K,K]')
    out_n = n + k.shape[-1] - 1 if c['deblur']['conv_type'] == 'full' else n
    if y.shape != (x.shape[0], 3, out_n, out_n):
        raise ValueError(f'y has shape {tuple(y.shape)}; expected [B,3,{out_n},{out_n}]. No resizing allowed.')
    if 'clean' in tensors and tensors['clean'].shape != x.shape:
        raise ValueError('clean shape must match x')
    for name, value in tensors.items():
        if not torch.isfinite(value).all():
            raise ValueError(f'{name} contains nonfinite values')
    save_bundle(a.output, **tensors, beta=linear_betas(c), settings=settings_tensor(c))
    print(f'Saved exact input bundle: batch={x.shape[0]}, FP32, no preprocessing')
