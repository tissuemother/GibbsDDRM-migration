"""One-time FP32 U-Net checkpoint -> TorchScript conversion; inference is C++."""
import argparse
import sys
from pathlib import Path
import torch
from common import ROOT, load_config
sys.path.insert(0, str(ROOT / 'vendor/gibbsddrm'))
from guided_diffusion.script_util import create_model
from guided_diffusion import unet

class ExportedDenoiser(torch.nn.Module):
    __constants__ = ['batch', 'image_size']
    def __init__(self, denoiser, batch: int, image_size: int):
        super().__init__()
        self.denoiser = denoiser
        self.batch = batch
        self.image_size = image_size
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.size(0) == self.batch and x.size(1) == 3, 'Export batch/channels mismatch'
        assert x.size(2) == self.image_size and x.size(3) == self.image_size, 'Export image size mismatch'
        assert t.dim() == 1 and t.size(0) == self.batch, 'Export timestep shape mismatch'
        return self.denoiser(x, t)

def export_model(config, checkpoint, output, batch=2):
    torch.backends.cudnn.benchmark = False
    if config['model']['use_fp16'] or config['model']['class_cond']:
        raise ValueError('This migration baseline requires FP32, unconditional inference')
    model = create_model(**config['model']).cpu().eval()
    # State dictionaries only; no executable checkpoint objects.
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    n = config['data']['image_size']
    x = torch.randn(batch, 3, n, n)
    t = torch.full((batch,), 990.0)
    # AttentionBlock hardcodes checkpoint(..., True), even when use_checkpoint
    # is false. The custom autograd Function is not traceable. For inference
    # only, execute the same forward function directly; preserve vendor files.
    original_checkpoint = unet.checkpoint
    def inference_checkpoint(function, inputs, parameters, flag):
        if torch.is_grad_enabled():
            raise RuntimeError('Export checkpoint adapter is inference-only')
        return function(*inputs)
    unet.checkpoint = inference_checkpoint
    try:
        with torch.no_grad():
            # The upstream UNet isn't scriptable without source changes. Trace tensor
            # execution while checking other timestep VALUES, including the final 0.
            traced = torch.jit.trace(model, (x, t), check_inputs=[
                (x * 0.5, torch.zeros(batch)), (x * -0.2, torch.full((batch,), 500.0))], strict=True)
            for step in (0, 10, 500, 990):
                tt = torch.full((batch,), float(step))
                torch.testing.assert_close(traced(x, tt), model(x, tt), rtol=1e-4, atol=1e-5)
    finally:
        unet.checkpoint = original_checkpoint
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    exported = torch.jit.script(ExportedDenoiser(traced, batch, n))
    exported.save(str(output))
    print(f'Saved {output}: FP32, batch={batch}, image_size={n}, benchmark=False')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT / 'configs/ffhq_256_deblur.yml'))
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch', type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(2)
    export_model(load_config(a.config), a.checkpoint, a.output, a.batch)
