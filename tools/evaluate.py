"""Evaluate final C++ output against a reference restoration / clean image."""
import argparse
import torch
import numpy as np
from PIL import Image

def images(tensor):
    return ((tensor + 1) / 2).clamp(0, 1)

def compare(actual, expected):
    a, b = images(actual), images(expected)
    if a.shape != b.shape:
        raise ValueError(f'Shape mismatch: {a.shape} != {b.shape}')
    mse = (a - b).square().flatten(1).mean(1)
    return {'mse_per_image': mse.tolist(), 'psnr_per_image_db': (-10 * torch.log10(mse)).tolist(),
            'max_abs_pixel_difference': (a - b).abs().max().item()}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', required=True)
    p.add_argument('--inputs', help='Input bundle with optional clean image')
    p.add_argument('--reference', help='Python final restoration .npy in [-1,1]')
    p.add_argument('--png-dir')
    a = p.parse_args()
    result = torch.jit.load(a.result).image
    if a.inputs:
        source = torch.jit.load(a.inputs)
        if hasattr(source, 'clean'):
            print('clean:', compare(result, source.clean))
    if a.reference:
        print('reference:', compare(result, torch.from_numpy(np.load(a.reference, allow_pickle=False))))
    if a.png_dir:
        from pathlib import Path
        Path(a.png_dir).mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images(result)):
            pixels = (image.permute(1, 2, 0) * 255).round().byte().numpy()
            Image.fromarray(pixels).save(Path(a.png_dir) / f'restored_{i}.png')
