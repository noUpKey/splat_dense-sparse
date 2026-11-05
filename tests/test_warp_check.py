"""Training-free warp check of the water-aware residual idea (uwsplat/eval/warp_check.py).

python -m tests.test_warp_check      (CPU, about a minute, no data or network weights needed)

Toy scene: a textured plane in water, seen by four training cameras and one test camera that is farther away than
both of its temporal neighbours (so the water term does not cancel between them). The photos are the *textured*
model under known per-frame affine colour corrections; the checkpoint holds the same geometry with *flat* colours
(a model that lost the texture). Expected, and asserted:
  - inverse_affine undoes apply_affine;
  - camera + water transfer (c) reproduces the test photo almost exactly, and beats camera-only (b) by a wide margin;
  - camera transfer (b) beats the naive copy (a), whose colour correction is the source frame's;
  - (c) beats the render (d), which has no texture, on PSNR and on high-frequency correlation;
  - render + water-scaled residual (e) is as good as (c) and beats the unscaled residual (f);
  - the JSON, the montage and the pre-registered decision are written / computed.
"""
import math
import os
import tempfile

import numpy as np
import torch

from tests.test_uncertainty import H, W, F, look_view
from uwsplat.common.io import save_json, save_rgb
from uwsplat.eval.heldout import apply_affine, centred_appearance, temporal_neighbors, blend
from uwsplat.eval.uncertainty import render_variant
from uwsplat.gs.medium import WaterMedium
import uwsplat.eval.warp_check as wc

C0 = 0.28209479177387814


def plane_params(textured: bool):
    xs, ys = np.arange(-4.2, 4.21, 0.14), np.arange(-3.2, 3.21, 0.14)
    pts = np.array([[x, y, 5.0] for x in xs for y in ys], dtype=np.float32)
    n = len(pts)
    if textured:
        col = np.stack([0.5 + 0.3 * np.sin(3.1 * pts[:, 0]) * np.cos(2.3 * pts[:, 1]),
                        0.5 + 0.3 * np.cos(2.7 * pts[:, 0] + 1.0),
                        0.5 + 0.3 * np.sin(1.9 * pts[:, 1] - 0.5 * pts[:, 0])], 1)
    else:
        col = np.full((n, 3), 0.5)
    return {"means": torch.tensor(pts),
            "scales": torch.full((n, 3), math.log(0.1)),
            "quats": torch.tensor([[1.0, 0, 0, 0]] * n),
            "opacities": torch.full((n,), math.log(0.99 / 0.01)),
            "sh0": ((torch.tensor(col, dtype=torch.float32) - 0.5) / C0)[:, None]}


def main():
    torch.manual_seed(0)
    # inverse_affine undoes apply_affine
    x = torch.rand(10, 12, 3)
    dA, db = 0.1 * torch.randn(3, 3), 0.05 * torch.randn(3)
    assert torch.allclose(wc.inverse_affine(apply_affine(x, dA, db), dA, db), x, atol=1e-5)

    try:
        wc.lpips(torch.rand(16, 16, 3), torch.rand(16, 16, 3))
    except Exception:
        wc.lpips = lambda a, b, net="vgg": 0.0          # no network weights offline

    true_p, flat_p = plane_params(True), plane_params(False)
    medium = WaterMedium(np.array([0.25, 0.35, 0.45]), np.array([0.20, 0.30, 0.40]), np.array([0.25, 0.20, 0.15]), 1)
    # frame order (= file order): train, train, TEST, train, train; the test camera is the farthest
    zs = [0.3, 0.0, -2.0, -0.5, -0.2]
    splits = ["train", "train", "test", "train", "train"]
    views = [look_view([0.05 * i, 0.0, z], target=(0.05 * i, 0.0, 5.0), split=s, idx=i)
             for i, (z, s) in enumerate(zip(zs, splits))]
    train_ids = [i for i, s in enumerate(splits) if s == "train"]
    for t, i in enumerate(train_ids):
        views[i].train_id = t
    A = 0.08 * torch.randn(len(train_ids), 3, 3)
    b = 0.04 * torch.randn(len(train_ids), 3)
    dAc, dbc = centred_appearance({"A": A, "b": b}, len(train_ids), "cpu")

    with tempfile.TemporaryDirectory() as d:
        images = []
        for v in views:
            img = render_variant(true_p, v, medium, 0, "torch", None, "base")["image"]
            if v.split == "train":
                img = apply_affine(img, dAc[v.train_id], dbc[v.train_id])
            else:  # the test frame's camera lies on the interpolation between its temporal neighbours
                img = apply_affine(img, *blend(temporal_neighbors(v.idx, train_ids), dAc, dbc))
            save_rgb(os.path.join(d, "images", v.stem + ".png"), (img.clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8))
            R, t = v.viewmat[:3, :3].numpy(), v.viewmat[:3, 3].numpy()
            images.append({"name": v.name, "file": f"images/{v.stem}.png", "registered": True, "split": v.split,
                           "R": R.tolist(), "t": t.tolist()})
        cam = {"model": "PINHOLE", "width": W, "height": H, "fx": F, "fy": F, "cx": W / 2, "cy": H / 2}
        save_json(os.path.join(d, "sfm", "scene.json"), {"scene": "toy", "camera": cam, "images": images})
        os.makedirs(os.path.join(d, "gs", "toy"))
        torch.save({"params": {k: v.clone() for k, v in flat_p.items()},
                    "cfg": {"downscale": 1.0, "backend": "torch", "sh_degree": 0},
                    "medium": medium.state_dict(), "bg": None, "appearance": {"A": A, "b": b}},
                   os.path.join(d, "gs", "toy", "ckpt.pt"))
        res = wc.run(d, "toy", device="cpu", backend="torch")
        s = res["summary"]
        m = s["masked"]
        print({k: (round(v["psnr"], 2), round(v["hf_corr"], 3)) for k, v in m.items()})
        assert s["mask_fraction"] > 0.3, s["mask_fraction"]
        assert m["camera_water"]["psnr"] > 35, m
        assert m["camera_water"]["psnr"] > m["camera"]["psnr"] + 5, m
        # with temporal interpolation the target camera is a blend of the sources' cameras, so (b) ~ (a) here;
        # the camera transfer is checked in the identity-colour run below
        assert m["camera"]["psnr"] > m["naive"]["psnr"] - 0.1, m
        assert m["camera_water"]["psnr"] > m["render"]["psnr"] + 5, m
        assert m["camera_water"]["hf_corr"] > m["render"]["hf_corr"] + 0.3, m
        assert m["render_res_water"]["psnr"] > m["render_res"]["psnr"] + 3, m
        assert abs(m["render_res_water"]["psnr"] - m["camera_water"]["psnr"]) < 3, m
        assert s["rel_range_change_median"] > 0.2, s
        assert os.path.exists(os.path.join(d, "eval", "toy", "warp_check.json"))
        assert os.path.exists(os.path.join(d, "eval", "toy", "warp_check", "f002.jpg"))
        dec = wc.decide({"toy": s})
        assert dec["physics"] and dec["detail_psnr"] and dec["build_network"], dec

        # overlap source selection + identity target colour runs too (photo-collection mode)
        res2 = wc.run(d, "toy", device="cpu", backend="torch", sources="overlap", target_colour="identity",
                      save_images=False)
        m2 = res2["summary"]["masked"]
        assert res2["summary"]["n_views"] == 1
        assert m2["camera"]["psnr"] > m2["naive"]["psnr"] + 3, m2      # source camera removed
    print("warp check test passed")


if __name__ == "__main__":
    main()
