"""Pseudo-view regulariser (gs/pseudo.py) and needle penalty (gs/losses.py).

python -m tests.test_pseudo      (CPU, seconds)
"""
import math

import torch

from uwsplat.gs.losses import needle_penalty
from uwsplat.gs.pseudo import interpolate_pose, points_depth, pseudo_depth_loss


def _vm(yaw, C):
    c, s = math.cos(yaw), math.sin(yaw)
    R = torch.tensor([[c, 0, -s], [0, 1.0, 0], [s, 0, c]])
    vm = torch.eye(4)
    vm[:3, :3], vm[:3, 3] = R, -R @ torch.tensor(C)
    return vm


def main():
    # pose interpolation: endpoints reproduced, midpoint halfway in centre and angle
    a, b = _vm(0.0, [0.0, 0, 0]), _vm(0.4, [1.0, 0, 0])
    for t, ref in ((0.0, a), (1.0, b)):
        vm, _ = interpolate_pose(a, b, t)
        assert torch.allclose(vm, ref, atol=1e-5)
    vm, C = interpolate_pose(a, b, 0.5)
    assert torch.allclose(C, torch.tensor([0.5, 0, 0])) and torch.allclose(vm, _vm(0.2, [0.5, 0, 0]), atol=1e-5)

    # z-buffer: a front plane at z=2 with a gap, a back plane at z=4 everywhere
    W = H = 40
    K = torch.tensor([[40.0, 0, 20], [0, 40.0, 20], [0, 0, 1]])
    g = torch.stack(torch.meshgrid(torch.linspace(-0.49, 0.49, 120), torch.linspace(-0.49, 0.49, 120),
                                   indexing="ij"), -1).reshape(-1, 2)
    front = torch.cat([g * 2, torch.full((len(g), 1), 2.0)], 1)
    hole = (front[:, 0].abs() < 0.08) & (front[:, 1].abs() < 0.08)       # a ~3 px gap in the front surface
    back = torch.cat([g * 4, torch.full((len(g), 1), 4.0)], 1)
    D = points_depth(torch.cat([front[~hole], back]), torch.eye(4), K, W, H)
    known = D > 0
    assert (D[known] < 2.1).all(), "a back-surface point seen through a small gap must be dropped"
    assert known.float().mean() > 0.8, known.float().mean()

    # the loss: zero for the right depth, |log ratio| for a wrong one, and only where a target exists
    alpha = torch.ones(H, W, 1)
    l0, cov = pseudo_depth_loss(D[..., None].clamp_min(1e-3), alpha, D)
    l1, _ = pseudo_depth_loss(torch.full((H, W, 1), 2.2), alpha, D)
    assert float(l0) < 1e-6 and abs(float(l1) - math.log(1.1)) < 1e-4 and 0.8 < float(cov) <= 1

    # needle penalty: zero for spheres and discs, positive (with gradient) for needles only
    s = torch.log(torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.01], [50.0, 1.0, 1.0]])).requires_grad_(True)
    p = needle_penalty(s, 10.0)
    p.backward()
    assert abs(float(p) - math.log(5.0) / 3) < 1e-5
    assert s.grad[:2].abs().sum() == 0 and s.grad[2, 0] > 0
    print(f"pseudo-view regulariser test passed (target coverage {float(known.float().mean()):.2f})")


if __name__ == "__main__":
    main()
