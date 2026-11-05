"""Variant composition ('mvs_lpips+abs') and AbsGS gradients in the CPU reference rasteriser.

python -m tests.test_variants      (CPU, a few seconds)
"""
import torch

from uwsplat.gs.raster import torch_rasterize
from uwsplat.gs.strategy import DensifyStrategy
from uwsplat.gs.trainer import TrainConfig
from uwsplat.pipeline import VARIANTS, _merge, apply_overrides, resolve_variant


def main():
    assert resolve_variant("mvs_lpips") == VARIANTS["mvs_lpips"]
    assert resolve_variant("mvs_lpips+abs") == {"geometry": "mvs", "w_lpips": 0.05,
                                                 "densify": {"absgrad": True, "grow_grad2d": 0.0008}}
    # densify dicts merge, later parts win
    assert resolve_variant("baseline_3dgs+abs")["densify"] == {"water_grad_scale": 1.0, "absgrad": True,
                                                               "grow_grad2d": 0.0008}
    assert resolve_variant("mvs_lpips+lpipsx2")["w_lpips"] == 0.1
    assert _merge(resolve_variant("mvs+abs"), {"densify": {"grow_grad2d": 0.0006}, "iterations": 7})["densify"] == \
        {"absgrad": True, "grow_grad2d": 0.0006}
    try:
        resolve_variant("mvs_lpips+nope")
        raise AssertionError("unknown part must fail")
    except KeyError:
        pass
    # every variant, and the batch-3 combinations, is a valid TrainConfig + DensifyStrategy
    for tag in list(VARIANTS) + ["mvs_lpips+abs", "mvs_lpips+prior", "mvs_lpips+surfel", "mvs_lpips+abs+prior"]:
        cfg = apply_overrides(TrainConfig(), resolve_variant(tag))
        DensifyStrategy(**cfg.densify)

    # absgrad (AbsGS): per Gaussian, sum over pixels of |d loss_pixel / d means2d|, checked against brute force
    torch.manual_seed(0)
    n, W, H = 4, 9, 7
    means = torch.cat([torch.rand(n, 2) * 1.0 - 0.5, torch.full((n, 1), 3.0)], 1).requires_grad_(True)
    quats = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
    scales = torch.full((n, 3), 0.25)
    opac = torch.full((n,), 0.8)
    colors = torch.rand(1, n, 3)
    K = torch.tensor([[[6.0, 0, W / 2], [0, 6.0, H / 2], [0, 0, 1]]])
    out, alpha, meta = torch_rasterize(means, quats, scales, opac, colors, torch.eye(4)[None], K, W, H, absgrad=True)
    wts = torch.rand(H, W, 3)
    m2d = meta["means2d"]
    per_pixel = (out[0, ..., :3] * wts).sum(-1)  # (H, W)
    brute = torch.zeros(n, 2)
    for y in range(H):
        for x in range(W):
            g, = torch.autograd.grad(per_pixel[y, x], m2d, retain_graph=True)
            brute += g[0].abs()
    total, = torch.autograd.grad(per_pixel.sum(), m2d, retain_graph=True)
    per_pixel.sum().backward()   # what training does; the hook stores means2d.absgrad
    assert hasattr(m2d, "absgrad") and torch.allclose(m2d.absgrad[0], brute, atol=1e-5), (m2d.absgrad, brute)
    assert (m2d.absgrad[0] >= total[0].abs() - 1e-6).all()   # |sum| <= sum |.|
    print(f"variant composition and absgrad test passed (|absgrad| / |grad| = "
          f"{float(m2d.absgrad.norm() / total.norm()):.2f})")


if __name__ == "__main__":
    main()
