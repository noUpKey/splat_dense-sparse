"""Forward scattering (gs/scatter.py): no-op at kappa -> 0, energy preserved, blur grows with range, gradients.

python -m tests.test_scatter      (CPU, seconds)
"""
import torch

from uwsplat.gs.raster import torch_rasterize
from uwsplat.gs.scatter import ForwardScatter


def _render(means, quats, scales, opac, K, W, H):
    colors = torch.ones(1, len(means), 3)
    out, alpha, _ = torch_rasterize(means, quats, scales, opac, colors, torch.eye(4)[None], K[None], W, H)
    return alpha[0, ..., 0]


def main():
    W, H = 160, 120
    K = torch.tensor([[150.0, 0, W / 2], [0, 150.0, H / 2], [0, 0, 1]])
    quats = torch.tensor([[1.0, 0, 0, 0]])
    scales = torch.tensor([[0.02, 0.02, 0.02]])
    opac = torch.tensor([0.3])
    vm = torch.eye(4)

    # kappa -> 0: nothing changes
    fs0 = ForwardScatter(1e-9)
    m = torch.tensor([[0.0, 0.0, 2.0]])
    s0, o0 = fs0.apply(m, quats, scales, opac, vm, K)
    assert torch.allclose(s0, scales, atol=1e-7) and torch.allclose(o0, opac, atol=1e-6)

    # energy: the summed alpha of an isolated, unsaturated Gaussian stays the same while its footprint grows
    fs = ForwardScatter(0.01)
    s1, o1 = fs.apply(m, quats, scales, opac, vm, K)
    a0, a1 = _render(m, quats, scales, opac, K, W, H), _render(m, quats, s1, o1, K, W, H)
    area = lambda a: float((a > a.max() / 2).sum())
    assert area(a1) > 1.5 * area(a0), (area(a0), area(a1))
    assert abs(float(a1.sum()) / float(a0.sum()) - 1) < 0.05, (float(a0.sum()), float(a1.sum()))

    # blur grows with range: same Gaussian at 1 and 3 units, angular width ratio ~ r
    near, far = torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[0.0, 0.0, 3.0]])
    bn = fs.apply(near, quats, torch.full((1, 3), 1e-6), opac, vm, K)[0][0, 0] / 1.0   # angular size ~ s / z
    bf = fs.apply(far, quats, torch.full((1, 3), 1e-6), opac, vm, K)[0][0, 0] / 3.0
    assert abs(float(bf / bn) - 3.0) < 0.05, float(bf / bn)

    # gradients reach kappa and the Gaussian parameters
    sc = scales.clone().requires_grad_(True)
    s2, o2 = fs.apply(m, quats, sc, opac, vm, K)
    (s2.sum() + o2.sum()).backward()
    assert fs.kappa_raw.grad is not None and sc.grad is not None
    print(f"forward scatter test passed (footprint x{area(a1) / area(a0):.1f}, "
          f"summed alpha ratio {float(a1.sum()) / float(a0.sum()):.3f})")


if __name__ == "__main__":
    main()
