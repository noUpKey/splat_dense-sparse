"""PSNR / SSIM / LPIPS, computed like the 3DGS / RUSplatting evaluation scripts.

Renders are quantised to 8 bits first (those scripts score saved PNGs), SSIM uses
the 3DGS Gaussian-window implementation, LPIPS uses the VGG network.
"""
import torch

from ..gs.losses import ssim as _ssim

_LPIPS = {}


def quantize(img: torch.Tensor) -> torch.Tensor:
    return torch.round(img.clamp(0, 1) * 255.0) / 255.0


def psnr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor = None) -> float:
    se = (a - b) ** 2
    mse = se[mask].mean() if mask is not None else se.mean()
    return float(-10.0 * torch.log10(mse.clamp_min(1e-12)))


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(_ssim(a, b))


def lpips(a: torch.Tensor, b: torch.Tensor, net: str = "vgg") -> float:
    if net not in _LPIPS:
        import lpips as lp
        _LPIPS[net] = lp.LPIPS(net=net, verbose=False).to(a.device).eval()
    f = _LPIPS[net]
    to = lambda x: x.permute(2, 0, 1)[None] * 2 - 1
    with torch.no_grad():
        return float(f(to(a), to(b)))
