"""Held-out views: which colour correction a test view gets, and which of its pixels any training view saw.

Training gives every training image its own affine colour correction (gs/appearance.py: I' = (Id + dA_i) I + db_i,
centred so that the average correction is the identity). A test view's correction is unknown, and evaluate()
renders it with the identity, i.e. the training average. Submerged3D frames change brightness by 6-16 % from one
frame to the next (auto exposure, strobes), so the average can be far from a given test frame. This module re-scores
a trained model under several choices of the test-view correction, none of which looks at test pixels:

  identity      the training average (what evaluate() reports)
  neighbors     linear interpolation in time between the nearest training frame before and after the view (frame
                order = sorted file names); at a sequence end, the nearest training frame
  nearest_pose  the training camera whose centre is nearest to the view's centre

plus `oracle`, an affine colour map fitted to the test image itself. The oracle uses the test pixels, so it is a
diagnostic bound (how much of the error is a global colour error), never a result. The same predictions are
checked on the training views leave-one-out (each training view predicted from the other training views), which
validates a choice without touching the test set.

Every test pixel is also marked seen / never seen: seen if, at its rendered depth (or as a direction at infinity
where the render is open water), it projects inside at least one training image. Occlusion is ignored, so `seen`
is an upper bound; the never-seen strip of a sequence-end view is what it is meant to find.
"""
import os

import numpy as np
import torch

from ..common.io import log, save_json
from ..gs.data import SceneData
from ..gs.render import render
from .evaluate import load_model
from .metrics import lpips, psnr, quantize, ssim

MODES = ("identity", "neighbors", "nearest_pose")


def centred_appearance(state, n_train: int, device):
    """(dA (n, 3, 3), db (n, 3)) of the training views, centred like Appearance; zeros if there is no model."""
    if state is None:
        return torch.zeros(n_train, 3, 3, device=device), torch.zeros(n_train, 3, device=device)
    A, b = state["A"].to(device), state["b"].to(device)
    return A - A.mean(0, keepdim=True), b - b.mean(0, keepdim=True)


def apply_affine(img: torch.Tensor, dA: torch.Tensor, db: torch.Tensor) -> torch.Tensor:
    """img (H, W, 3) -> (Id + dA) img + db, exactly as Appearance.forward."""
    return img @ (torch.eye(3, device=img.device) + dA).T + db


def temporal_neighbors(idx: int, train_idx: list, exclude: int = None) -> list:
    """[(position in train_idx, weight)]: linear interpolation in frame order between the nearest training frame
    before and after `idx`; one frame at a sequence end. `exclude` (a position) is left out (leave-one-out)."""
    cand = [(p, i) for p, i in enumerate(train_idx) if p != exclude]
    before = max(((i, p) for p, i in cand if i < idx), default=None)
    after = min(((i, p) for p, i in cand if i > idx), default=None)
    if before is None or after is None:
        if before is None and after is None:
            return []
        i, p = before or after
        return [(p, 1.0)]
    (i0, p0), (i1, p1) = before, after
    return [(p0, (i1 - idx) / (i1 - i0)), (p1, (idx - i0) / (i1 - i0))]


def nearest_pose(campos: torch.Tensor, centres: torch.Tensor, exclude: int = None) -> list:
    d = (centres - campos).norm(dim=-1)
    if exclude is not None:
        d[exclude] = float("inf")
    return [(int(torch.argmin(d)), 1.0)]


def blend(weights: list, dA: torch.Tensor, db: torch.Tensor):
    A = sum(w * dA[p] for p, w in weights)
    b = sum(w * db[p] for p, w in weights)
    return A, b


def oracle_affine(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Least-squares affine colour map pred -> gt (3x3 + offset, fitted on all pixels): a diagnostic bound only."""
    X = torch.cat([pred.reshape(-1, 3), torch.ones_like(pred.reshape(-1, 3)[:, :1])], 1).double()
    Y = gt.reshape(-1, 3).double()
    M = torch.linalg.solve(X.T @ X + 1e-9 * torch.eye(4, dtype=X.dtype, device=X.device), X.T @ Y)
    return (X @ M).float().reshape(pred.shape)


def seen_mask(view, depth: torch.Tensor, alpha: torch.Tensor, train_views, alpha_min: float = 0.5) -> torch.Tensor:
    """(H, W) bool: the pixel's rendered surface point (or, where alpha <= alpha_min, its direction at infinity)
    projects inside at least one training image (occlusion ignored)."""
    H, W = view.hw
    dev = depth.device
    v, u = torch.meshgrid(torch.arange(H, device=dev) + 0.5, torch.arange(W, device=dev) + 0.5, indexing="ij")
    K = view.K
    rays = torch.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], torch.ones_like(u)], -1)  # z = 1
    R, t = view.viewmat[:3, :3], view.viewmat[:3, 3]
    surface = alpha[..., 0] > alpha_min
    Xw = (rays * depth[..., :1] - t) @ R          # camera -> world for row vectors: R^T (x - t)
    Dw = rays @ R                                 # direction only (point at infinity)
    seen = torch.zeros(H, W, dtype=torch.bool, device=dev)
    for tv in train_views:
        Rj, tj, Kj = tv.viewmat[:3, :3], tv.viewmat[:3, 3], tv.K
        P = torch.where(surface[..., None], Xw @ Rj.T + tj, Dw @ Rj.T)
        z = P[..., 2]
        zs = torch.where(z > 1e-6, z, torch.ones_like(z))
        uj = Kj[0, 0] * P[..., 0] / zs + Kj[0, 2]
        vj = Kj[1, 1] * P[..., 1] / zs + Kj[1, 2]
        Hj, Wj = tv.hw
        seen |= (z > 1e-6) & (uj >= 0) & (uj < Wj) & (vj >= 0) & (vj < Hj)
    return seen


def _scores(pred, gt, seen=None, lpips_net="vgg", full=True):
    q = quantize(pred)
    m = {"psnr": psnr(q, gt)}
    if full:
        m["ssim"], m["lpips"] = ssim(q, gt), lpips(q, gt, lpips_net)
    if seen is not None:
        if seen.any():
            m["psnr_seen"] = psnr(q, gt, seen)
        if (~seen).any():
            m["psnr_unseen"] = psnr(q, gt, ~seen)
    return m


@torch.no_grad()
def check_heldout(out_dir: str, tag: str, device: str = "cuda", backend: str = None, lpips_net: str = "vgg") -> dict:
    ckpt_path = os.path.join(out_dir, "gs", tag, "ckpt.pt")
    params, medium, bg, cfg = load_model(ckpt_path, device)
    app_state = torch.load(ckpt_path, map_location=device, weights_only=False).get("appearance")
    backend = backend or cfg["backend"]
    data = SceneData(out_dir, device, cfg["downscale"], load_priors=False)
    train = sorted(data.train, key=lambda v: v.train_id)
    train_idx = [v.idx for v in train]
    centres = torch.stack([v.campos for v in train])
    dA, db = centred_appearance(app_state, len(train), device)
    name = lambda p: train[p].stem

    def predict(view, exclude=None):
        ws = {"identity": [], "neighbors": temporal_neighbors(view.idx, train_idx, exclude),
              "nearest_pose": nearest_pose(view.campos, centres, exclude)}
        return {k: (blend(w, dA, db) if w else (torch.zeros(3, 3, device=device), torch.zeros(3, device=device)), w)
                for k, w in ws.items()}

    # training views, leave-one-out: does a prediction beat the identity where the true correction is known?
    train_res = {}
    for p, v in enumerate(train):
        img = render(params, v, medium, cfg["sh_degree"], backend, bg_color=bg)["image"]
        r = {k: _scores(apply_affine(img, *ab), v.image, full=False)["psnr"] for k, (ab, _) in predict(v, p).items()}
        r["own"] = _scores(apply_affine(img, dA[p], db[p]), v.image, full=False)["psnr"]
        r["strength"] = psnr(apply_affine(img, dA[p], db[p]).clamp(0, 1), img.clamp(0, 1))  # identity vs own, dB
        train_res[v.name] = r

    test_res = {}
    for v in data.test:
        out = render(params, v, medium, cfg["sh_degree"], backend, bg_color=bg)
        img = out["image"]
        seen = seen_mask(v, out["depth"], out["alpha"], train)
        r = {"frame": v.idx, "seen_fraction": float(seen.float().mean())}
        for k, ((A, b), w) in predict(v).items():
            r[k] = _scores(apply_affine(img, A, b), v.image, seen, lpips_net)
            if w:
                r[k]["from"] = [[name(p), round(float(wt), 3)] for p, wt in w]
        r["oracle"] = _scores(oracle_affine(img, v.image), v.image, seen, lpips_net)
        test_res[v.name] = r

    modes = MODES + ("oracle",)
    mean = {k: {m: float(np.mean([r[k][m] for r in test_res.values() if m in r[k]]))
                for m in ("psnr", "ssim", "lpips", "psnr_seen", "psnr_unseen")
                if any(m in r[k] for r in test_res.values())} for k in modes}
    train_mean = {k: float(np.mean([r[k] for r in train_res.values()])) for k in MODES + ("own", "strength")}
    res = {"scene": data.meta["scene"], "tag": tag, "lpips_net": lpips_net, "has_appearance": app_state is not None,
           "test": {"mean": mean, "per_view": test_res}, "train_leave_one_out": {"mean": train_mean,
                                                                              "per_view": train_res}}
    save_json(os.path.join(out_dir, "eval", tag, "heldout_appearance.json"), res)
    log(f"[{tag}] {data.meta['scene']} test PSNR / SSIM / LPIPS: " + " | ".join(
        f"{k} {mean[k]['psnr']:.2f} / {mean[k]['ssim']:.4f} / {mean[k]['lpips']:.4f}" for k in modes))
    log(f"[{tag}] {data.meta['scene']} train PSNR, leave-one-out: " + " | ".join(
        f"{k} {train_mean[k]:.2f}" for k in MODES + ("own",)) + f" (correction strength {train_mean['strength']:.1f} dB)")
    return res
