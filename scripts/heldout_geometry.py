"""Held-out check of the dense geometry, before any splatting: does it explain the *test* images?

python scripts/heldout_geometry.py --out outputs/Isro [--neighbours 2] [--perturb 0.06]

For every held-out test view and its nearest training views (by shared SIFT points), the test image is
warped into the training view with a candidate depth map of that training view, and the windowed
zero-mean NCC (7x7, grey, MVS grid) between the training image and the warped test image is measured.
NCC is invariant to the per-window gain/offset that water and exposure add, so a high NCC means the
geometry puts the texture where an unseen camera sees it - what SSIM / LPIPS on test views reward.

Candidates (each on the training view's MVS grid, compared on the pixels where both exist):
  sift      : linear interpolation of the training SIFT keypoint depths inside the wreck mask
              (fusion v1 gives the wreck only SIFT points);
  network   : the dense network's fused depth (dense/, when the dense stage ran) - v1's dense geometry;
  verified  : mvs/ verified depth (dense cloud v2);
  perturbed : verified x a smooth random +-p field (p = --perturb), a network-sized error on v2's pixels.
Test images are used for scoring only, exactly like PSNR; nothing here feeds back into training.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uwsplat.common.io import load_json, load_rgb  # noqa: E402
from uwsplat.dense.predict import NetGrid  # noqa: E402
from uwsplat.dense.run import load_scene  # noqa: E402
from uwsplat.mvs.run import relative_pose, resample, to_grid  # noqa: E402
from uwsplat.mvs.sweep import masked_ncc, pixel_rays, window_std  # noqa: E402


def warp_ncc(ref, src, K, rel, depth, window=7):
    """NCC between ref and src warped into ref with `depth` (0 = none). All on the same grid."""
    h, w = ref.shape
    rays = pixel_rays(K, h, w)
    X = rel[:, :3] @ (rays * depth.reshape(1, -1)) + rel[:, 3:4]
    z = X[2]
    u = K[0, 0] * X[0] / z.clamp_min(1e-6) + K[0, 2]
    v = K[1, 1] * X[1] / z.clamp_min(1e-6) + K[1, 2]
    ok = (depth.reshape(-1) > 0) & (z > 1e-6) & (u >= 0) & (u <= w) & (v >= 0) & (v <= h)
    grid = torch.stack([u / w * 2 - 1, v / h * 2 - 1], -1).reshape(1, h, w, 2)
    warped = F.grid_sample(src[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return masked_ncc(ref[None, None], warped, ok.reshape(1, 1, h, w).float(), window)[0, 0]


def smooth_field(h, w, p, rng, cells=6):
    """Smooth multiplicative error exp(N(0, p)) varying over ~1/cells of the image."""
    f = rng.normal(0, p, (cells, int(round(cells * w / h))))
    f = cv2.resize(f.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    return np.exp(f * p / max(f.std(), 1e-9))


def main(out, neighbours=2, perturb=0.06, seed=0):
    meta, pts, R, t, train = load_scene(out)
    summ = load_json(os.path.join(out, "mvs", "summary.json"))
    g = NetGrid(**summ["grid"])
    Km = np.array(summ["K"])
    K = torch.tensor(Km, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    have_net = os.path.exists(os.path.join(out, "dense", "summary.json"))
    ng = NetGrid(**load_json(os.path.join(out, "dense", "summary.json"))["grid"]) if have_net else None
    stems = [os.path.splitext(im["name"])[0] for im in meta["images"]]
    tests = [i for i, im in enumerate(meta["images"]) if im["registered"] and im["split"] == "test"]
    views = [int(v) for v in np.where(train)[0]]
    seen = {v: set(pts["obs_pt"][pts["obs_img"] == v].tolist()) for v in range(len(meta["images"]))}
    gray = lambda i: torch.tensor(cv2.cvtColor(to_grid(load_rgb(os.path.join(out, meta["images"][i]["file"])), g),
                                               cv2.COLOR_RGB2GRAY), dtype=torch.float32)
    rows = []
    for tv in tests:
        near = sorted(views, key=lambda j: -len(seen[tv] & seen[j]))[:neighbours]
        src = gray(tv)
        for i in near:
            pr = np.load(os.path.join(out, "mvs", "priors", stems[i] + ".npz"))
            verified = np.where(pr["cls"] == 2, pr["dense_depth"], 0).astype(np.float32)
            wreck = cv2.imread(os.path.join(out, "mvs", "masks", stems[i] + ".png"), cv2.IMREAD_GRAYSCALE)
            wreck_g = resample((wreck > 127).astype(np.uint8), NetGrid(g.W, g.H, g.W, g.H, g.H, 1.0, 1.0, 0.0), g) > 0
            uv = pr["kp_uv"] * np.array([g.sx, g.sy])
            from scipy.interpolate import LinearNDInterpolator
            yy, xx = np.mgrid[0:g.h, 0:g.w]
            interp = LinearNDInterpolator(uv, pr["kp_z"], fill_value=0.0)
            sift = np.where(wreck_g, interp(xx + 0.5, yy + 0.5), 0).astype(np.float32)
            cands = {"sift": sift, "verified": verified,
                     "perturbed": verified * smooth_field(g.h, g.w, perturb, rng).astype(np.float32)}
            if have_net:
                dn = np.load(os.path.join(out, "dense", stems[i] + ".npz"))
                cands["network"] = resample(dn["depth"].astype(np.float32), ng, g)
            ref = gray(i)
            textured = (window_std(ref, 7) >= 2.0).numpy()
            rel = torch.tensor(relative_pose(R, t, i, tv), dtype=torch.float32)
            ncc = {k: warp_ncc(ref, src, K, rel, torch.from_numpy(v)).numpy() for k, v in cands.items()}
            row = {"test": meta["images"][tv]["name"], "train": meta["images"][i]["name"],
                   "shared_sift": len(seen[tv] & seen[i])}
            for k in cands:
                m = (ncc[k] > -1) & textured
                row[k] = {"coverage_textured": float(m.sum() / max(textured.sum(), 1)),
                          "mean_ncc": float(ncc[k][m].mean()) if m.any() else None}
            for a, b in (("sift", "verified"), ("network", "verified"), ("perturbed", "verified")):
                if a in ncc:
                    m = (ncc[a] > -1) & (ncc[b] > -1) & textured & wreck_g if a == "sift" else \
                        (ncc[a] > -1) & (ncc[b] > -1) & textured
                    if m.sum() > 100:
                        row[f"{a}_vs_{b}"] = {"n": int(m.sum()), a: float(ncc[a][m].mean()), b: float(ncc[b][m].mean()),
                                              f"{a}_frac_ncc>0.5": float((ncc[a][m] > 0.5).mean()),
                                              f"{b}_frac_ncc>0.5": float((ncc[b][m] > 0.5).mean())}
            rows.append(row)
            msg = "  ".join(f"{k}: {row[k]['mean_ncc']:.3f} on {100 * row[k]['coverage_textured']:.0f}%"
                            for k in cands if row[k]["mean_ncc"] is not None)
            print(f"{row['test']} <- {row['train']}: {msg}")
    pairs = {}
    for key in ("sift_vs_verified", "network_vs_verified", "perturbed_vs_verified"):
        sel = [r[key] for r in rows if key in r]
        if sel:
            n = np.array([s["n"] for s in sel], float)
            pairs[key] = {k: float(np.sum(n * np.array([s[k] for s in sel])) / n.sum())
                          for k in sel[0] if k != "n"}
            pairs[key]["pixels"] = int(n.sum())
    res = {"scene": meta["scene"], "pairs": rows, "same_pixels": pairs}
    with open(os.path.join(out, "mvs", "heldout_geometry.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(pairs, indent=1))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--neighbours", type=int, default=2)
    p.add_argument("--perturb", type=float, default=0.06)
    a = p.parse_args()
    main(a.out, a.neighbours, a.perturb)
