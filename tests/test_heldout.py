"""Held-out view check (uwsplat/eval/heldout.py): colour-correction predictions, oracle fit, seen mask.

python -m tests.test_heldout      (CPU, a few seconds, no data or network weights needed)
"""
import torch

from uwsplat.eval.heldout import apply_affine, blend, nearest_pose, oracle_affine, seen_mask, temporal_neighbors
from uwsplat.eval.metrics import psnr
from uwsplat.gs.appearance import Appearance
from uwsplat.gs.data import View


def _view(C, W=200, H=100, f=100.0, R=None):
    R = torch.eye(3) if R is None else R
    vm = torch.eye(4)
    vm[:3, :3], vm[:3, 3] = R, -R @ torch.tensor(C, dtype=torch.float32)
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return View(0, "v", "v", "test", vm, K, torch.tensor(C, dtype=torch.float32), torch.zeros(H, W, 3),
                torch.zeros(H, W, 3), torch.ones(H, W, 1))


def main():
    # frames 0..19, every 8th held out -> training frames 1-7, 9-15, 17-19
    train_idx = [i for i in range(20) if i % 8]
    pos = {i: p for p, i in enumerate(train_idx)}
    assert temporal_neighbors(0, train_idx) == [(pos[1], 1.0)]                       # sequence start
    assert temporal_neighbors(8, train_idx) == [(pos[7], 0.5), (pos[9], 0.5)]        # interior: halfway
    w = dict(temporal_neighbors(5, train_idx, exclude=pos[5]))                       # leave-one-out
    assert w == {pos[4]: 0.5, pos[6]: 0.5}
    w = dict(temporal_neighbors(7, train_idx, exclude=pos[7]))                       # gap: 6 and 9
    assert abs(w[pos[6]] - 2 / 3) < 1e-9 and abs(w[pos[9]] - 1 / 3) < 1e-9
    assert temporal_neighbors(19, train_idx, exclude=pos[19]) == [(pos[18], 1.0)]    # sequence end
    centres = torch.tensor([[float(i), 0.0, 0.0] for i in train_idx])
    assert nearest_pose(torch.tensor([8.2, 0.0, 0.0]), centres) == [(pos[9], 1.0)]
    assert nearest_pose(centres[pos[9]], centres, exclude=pos[9]) == [(pos[10], 1.0)]

    # the correction applied to test views is exactly the training model's (centred) correction
    torch.manual_seed(0)
    app = Appearance(len(train_idx))
    with torch.no_grad():
        app.A.normal_(0, 0.1)
        app.b.normal_(0, 0.05)
    dA, db = app.A - app.A.mean(0, keepdim=True), app.b - app.b.mean(0, keepdim=True)
    img = torch.rand(30, 40, 3)
    assert torch.allclose(apply_affine(img, dA[3], db[3]), app(img, 3), atol=1e-6)
    A, b = blend([(2, 0.25), (3, 0.75)], dA, db)
    assert torch.allclose(A, 0.25 * dA[2] + 0.75 * dA[3]) and torch.allclose(b, 0.25 * db[2] + 0.75 * db[3])

    # oracle: recovers an affine colour change exactly
    M, c = torch.eye(3) * 0.7 + 0.05 * torch.randn(3, 3), torch.tensor([0.1, -0.05, 0.08])
    gt = img @ M.T + c
    assert psnr(oracle_affine(img, gt), gt) > 80

    # seen mask, surface branch: a training camera at x=0 and a test camera at x=1, both looking at the plane z=5.
    # Test pixel u sees world x = 1 + 5 (u - 100) / 100; the training image covers x in [-5, 5) -> seen iff u < 180.
    train = [_view([0.0, 0.0, 0.0])]
    test = _view([1.0, 0.0, 0.0])
    depth, alpha = torch.full((100, 200, 1), 5.0), torch.ones(100, 200, 1)
    seen = seen_mask(test, depth, alpha, train)
    assert seen[:, :179].all() and not seen[:, 181:].any(), seen.float().mean()
    # open water (alpha 0): directions at infinity, a pure translation sees the same directions
    assert seen_mask(test, depth, torch.zeros_like(alpha), train).all()
    # a test camera turned by 30 degrees about y sees directions outside the training camera's 90-degree field
    a = torch.tensor(30.0).deg2rad()
    Ry = torch.tensor([[torch.cos(a), 0, -torch.sin(a)], [0, 1, 0], [torch.sin(a), 0, torch.cos(a)]])
    frac = float(seen_mask(_view([0.0, 0.0, 0.0], R=Ry), depth, torch.zeros_like(alpha), train).float().mean())
    assert 0.55 < frac < 0.85, frac
    print(f"held-out view check test passed (turned camera: {100 * frac:.0f}% of directions seen)")


if __name__ == "__main__":
    main()
