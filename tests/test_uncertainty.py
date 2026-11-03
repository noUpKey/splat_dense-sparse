"""Geometric-uncertainty splatting (uwsplat/eval/uncertainty.py).

python -m tests.test_uncertainty      (CPU, under a minute, no data or network weights needed)

Checks: covariance <-> (quaternion, scales) round trip; the 2D opacity factor conserves a splat's integral; the
measured blending weights sum to the rendered alpha; the training-view bound J P J^T <= sigma2 I; a Gaussian seen by
one view gets its long axis along that view's ray; a training view barely changes while an isolated splat's energy
is conserved in a new view; and the full experiment (load checkpoint -> P -> variants -> JSON) on a tiny scene.
"""
import json
import math
import os
import tempfile

import numpy as np
import torch

from uwsplat.common.io import save_json, save_rgb
from uwsplat.eval.uncertainty import (cov_to_quat_scale, factor_2d, gaussian_cov, position_uncertainty,
                                      render_variant, run, visibility_weights)
from uwsplat.gs.data import View
from uwsplat.gs.medium import WaterMedium

W, H, F = 64, 48, 60.0


def look_view(C, target=(0.0, 0.0, 5.0), split="train", train_id=-1, idx=0):
    C = torch.tensor(C, dtype=torch.float32)
    z = torch.tensor(target) - C
    z = z / z.norm()
    x = torch.linalg.cross(torch.tensor([0.0, 1.0, 0.0]), z)
    x = x / x.norm()
    y = torch.linalg.cross(z, x)
    R = torch.stack([x, y, z])                       # world -> camera
    vm = torch.eye(4)
    vm[:3, :3], vm[:3, 3] = R, -R @ C
    K = torch.tensor([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    v, u = torch.meshgrid(torch.arange(H) + 0.5, torch.arange(W) + 0.5, indexing="ij")
    rays = torch.stack([(u - W / 2) / F, (v - H / 2) / F, torch.ones_like(u)], -1)
    norm = rays.norm(dim=-1, keepdim=True)
    dirs = (rays / norm) @ R
    view = View(idx, f"f{idx:03d}.png", f"f{idx:03d}", split, vm, K, C, torch.zeros(H, W, 3), dirs, norm)
    view.train_id = train_id
    return view


def params_for(means, scale=0.06, opacity=0.9, colour=(0.8, 0.4, 0.2)):
    n = len(means)
    return {"means": torch.tensor(means, dtype=torch.float32),
            "scales": torch.full((n, 3), math.log(scale)),
            "quats": torch.tensor([[1.0, 0, 0, 0]] * n),
            "opacities": torch.full((n,), math.log(opacity / (1 - opacity))),
            "sh0": ((torch.tensor(colour) - 0.5) / 0.28209479177387814).repeat(n, 1)[:, None]}


def main():
    torch.manual_seed(0)
    # 1. covariance -> (quaternion, scales) -> covariance
    A = torch.randn(50, 3, 3)
    S = A @ A.transpose(1, 2) + 0.1 * torch.eye(3)
    q, s = cov_to_quat_scale(S)
    assert torch.allclose(gaussian_cov(q, s), S, atol=1e-4, rtol=1e-4)
    q7, s7 = cov_to_quat_scale(S, chunk=7)                         # chunked CPU path = one batch
    assert torch.allclose(gaussian_cov(q7, s7), S, atol=1e-4, rtol=1e-4)

    # 2. the 2D factor conserves the integral alpha * 2 pi sqrt(det(J S J^T + eps))
    J = torch.randn(50, 2, 3) * 10
    S2 = S + torch.diag_embed(torch.rand(50, 3)) * 5
    fac = factor_2d(J, S, S2)
    I2 = 0.3 * torch.eye(2)
    det = lambda c: torch.linalg.det(c)
    assert torch.allclose(det(J @ S @ J.transpose(1, 2) + I2).sqrt(), fac * det(J @ S2 @ J.transpose(1, 2) + I2).sqrt(),
                          rtol=1e-4)

    # scene: points on the plane z = 5; three training cameras; the rightmost points are seen by one camera only
    xs = np.linspace(-1.2, 1.2, 9)
    pts = [[x, y, 5.0] for x in xs for y in (-0.4, 0.0, 0.4)] + [[3.6, 0.0, 5.0]]
    p = params_for(pts)
    train = [look_view([cx, 0.0, 0.0], target=(cx, 0.0, 5.0), train_id=k, idx=1 + k) for k, cx in
             enumerate((-0.6, 0.0, 0.6))]
    train.append(look_view([3.0, 0.0, 0.0], target=(3.0, 0.0, 5.0), train_id=3, idx=4))   # only one sees x = 3.6

    # 3. blending weights from the gradient trick sum to the rendered alpha
    for v in train:
        Wk = visibility_weights(p, None, v, "torch")
        out, alpha, _ = __import__("uwsplat.gs.raster", fromlist=["rasterize"]).rasterize(
            p["means"], p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
            torch.zeros(1, len(pts), 1), v.viewmat[None], v.K[None], W, H, backend="torch")
        assert abs(float(Wk.sum()) - float(alpha.sum())) < 1e-3 * float(alpha.sum()) + 1e-4

    # 4. the bound, and the long axis of a single-view Gaussian lies along its ray
    P, mipvar, nviews, stats = position_uncertainty(p, None, train, "torch", sigma2=0.2)
    assert stats["bound_check_max_ratio"] <= 1.0 + 1e-3, stats
    assert int(nviews[-1]) == 1, nviews
    evals, evecs = torch.linalg.eigh(P[-1].double())
    ray = (p["means"][-1] - train[3].campos).double()
    ray = ray / ray.norm()
    assert abs(float(evecs[:, -1] @ ray)) > 0.99 and evals[-1] > 100 * evals[0], (evals, evecs[:, -1])

    # 5. a training view barely changes; a new view sees the single-view splat spread with its energy conserved
    sh = 0
    bg = torch.tensor([0.1, 0.3, 0.4])
    kw = dict(P=P, mipvar=mipvar)
    for v in train[:3]:
        a = render_variant(p, v, None, sh, "torch", bg, "base")["image"]
        b = render_variant(p, v, None, sh, "torch", bg, "c2", **kw)["image"]
        mse = float(((a - b) ** 2).mean())
        assert mse < 1e-4, mse                                     # > 40 dB
    lone = params_for([[3.6, 0.0, 5.0]], opacity=0.3)
    P1, _, _, _ = position_uncertainty(lone, None, [train[3]], "torch", sigma2=0.2)
    new = look_view([4.0, 0.0, 1.6], target=(3.6, 0.0, 5.0), split="test", idx=0)
    a0 = render_variant(lone, new, None, sh, "torch", bg, "base")["alpha"].sum()
    a1 = render_variant(lone, new, None, sh, "torch", bg, "c2", P=P1, mipvar=None)["alpha"].sum()
    peak0 = render_variant(lone, new, None, sh, "torch", bg, "base")["alpha"].max()
    peak1 = render_variant(lone, new, None, sh, "torch", bg, "c2", P=P1, mipvar=None)["alpha"].max()
    assert abs(float(a1 / a0) - 1) < 0.08 and float(peak1) < 0.7 * float(peak0), (a0, a1, peak0, peak1)

    # 6. full experiment on a tiny scene folder (checkpoint, scene.json, images) with a water medium.
    # LPIPS needs the VGG weights (a download); if they are not available offline, score it as 0 here
    # (LPIPS itself is covered by tests/test_perceptual.py).
    import uwsplat.eval.uncertainty as unc
    try:
        unc.lpips(torch.rand(16, 16, 3), torch.rand(16, 16, 3))
    except Exception:
        unc.lpips = lambda a, b, net="vgg": 0.0
    with tempfile.TemporaryDirectory() as d:
        views = train[:3] + [look_view([-1.2, 0.0, 0.0], target=(-1.2, 0.0, 5.0), split="test", idx=0),
                             look_view([0.3, 0.0, 0.0], target=(0.3, 0.0, 5.0), split="test", idx=2)]
        medium = WaterMedium(np.array([0.2, 0.3, 0.4]), np.array([0.3, 0.2, 0.1]), np.array([0.2, 0.2, 0.2]), 1)
        images = []
        for v in sorted(views, key=lambda v: v.idx):
            img = render_variant(p, v, medium, sh, "torch", None, "base")["image"]
            img = (img + 0.02 * torch.randn_like(img)).clamp(0, 1)
            save_rgb(os.path.join(d, "images", v.stem + ".png"), (img.numpy() * 255 + 0.5).astype(np.uint8))
            R, t = v.viewmat[:3, :3].numpy(), v.viewmat[:3, 3].numpy()
            images.append({"name": v.name, "file": f"images/{v.stem}.png", "registered": True, "split": v.split,
                           "R": R.tolist(), "t": t.tolist()})
        cam = {"model": "PINHOLE", "width": W, "height": H, "fx": F, "fy": F, "cx": W / 2, "cy": H / 2}
        save_json(os.path.join(d, "sfm", "scene.json"), {"scene": "toy", "camera": cam, "images": images})
        os.makedirs(os.path.join(d, "gs", "toy"))
        torch.save({"params": {k: v.clone() for k, v in p.items()},
                    "cfg": {"downscale": 1.0, "backend": "torch", "sh_degree": 0},
                    "medium": medium.state_dict(), "bg": None,
                    "appearance": {"A": torch.zeros(3, 3, 3), "b": torch.zeros(3, 3)}},
                   os.path.join(d, "gs", "toy", "ckpt.pt"))
        res = run(d, "toy", device="cpu", backend="torch", save_images=True)
        with open(os.path.join(d, "eval", "toy", "uncertainty_splat.json")) as f:
            assert json.load(f)["scene"] == "toy"
        positions = {r["position"] for r in res["test"].values()}
        assert positions == {"start", "interior"}, positions
        tr = res["summary"]["train_psnr"]
        assert tr["c2"] > tr["base"] - 0.5, tr                     # training views essentially unchanged
        assert res["uncertainty_stats"]["bound_check_max_ratio"] <= 1.0 + 1e-3
        assert os.listdir(os.path.join(d, "eval", "toy", "uncertainty"))
        print("uncertainty splatting test passed:",
              {k: round(v, 2) for k, v in tr.items()}, "train PSNR;",
              {k: round(res["summary"]["start"][k]["psnr"], 2) for k in ("base", "c2", "fade", "mip")}, "start view")


if __name__ == "__main__":
    main()
