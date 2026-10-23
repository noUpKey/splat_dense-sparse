"""Metric alignment and fusion of pairwise pointmaps using the SIFT SfM.

Replaces the reference method's global alignment (which optimises poses, focal
lengths and depth maps of every pair) with closed-form steps, since poses and
intrinsics are already known precisely from the sparse branch:
  1. per pair: depth = z of the reference pointmap; one robust scale factor
     maps it onto the depths of the SfM points seen in that view;
  2. per view: confidence-weighted log-depth fusion of its k pairs, keeping
     only predictions that agree with the per-pixel median;
  3. across views: a COLMAP-fusion-style geometric consistency count with the
     known poses. Water-column 'geometry' hallucinated by the network is not
     multi-view consistent and is rejected here.
"""
import numpy as np
import torch
import torch.nn.functional as F


def sample(img: torch.Tensor, uv: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """img (C, h, w) or (h, w); uv (n, 2) in +0.5 pixel convention -> (C, n)."""
    if img.dim() == 2:
        img = img[None]
    h, w = img.shape[-2:]
    g = torch.stack([uv[:, 0] / w * 2 - 1, uv[:, 1] / h * 2 - 1], -1)[None, None]
    return F.grid_sample(img[None].float(), g.float(), mode=mode, align_corners=False)[0, :, 0]


def fit_scale(d_pred: np.ndarray, z_sfm: np.ndarray, w: np.ndarray, inlier_tol: float = 0.1):
    """Robust scale s with z_sfm ~ s * d_pred (log-space median, then weighted inlier mean)."""
    ok = (d_pred > 1e-6) & (z_sfm > 1e-6) & np.isfinite(d_pred)
    if ok.sum() < 10:
        return np.nan, {"n": int(ok.sum())}
    lr = np.log(z_sfm[ok]) - np.log(d_pred[ok])
    s0 = np.median(lr)
    inl = np.abs(lr - s0) < np.log1p(inlier_tol)
    s = np.sum(w[ok][inl] * lr[inl]) / np.sum(w[ok][inl])
    rel = np.abs(np.expm1(lr - s))
    return float(np.exp(s)), {"n": int(ok.sum()), "inlier_ratio": float(inl.mean()),
                              "median_rel_err": float(np.median(rel))}


def fuse_view(depths: torch.Tensor, confs: torch.Tensor, tol: float = 0.05):
    """depths/confs (K, h, w) of one view -> fused depth, mean agreeing conf, #agreeing pairs."""
    logd = torch.log(depths.clamp_min(1e-8))
    valid = torch.isfinite(logd) & (depths > 0)
    med = torch.nanmedian(torch.where(valid, logd, torch.nan), dim=0).values
    agree = valid & ((logd - med).abs() < np.log1p(tol))
    w = confs * agree
    fused = torch.exp((w * torch.nan_to_num(logd)).sum(0) / w.sum(0).clamp_min(1e-8))
    n = agree.sum(0)
    conf = (confs * agree).sum(0) / n.clamp_min(1)
    fused[n == 0] = 0
    return fused, conf, n


def cross_view_consistency(D: torch.Tensor, Kn: np.ndarray, R: np.ndarray, t: np.ndarray, tol: float = 0.05):
    """D (V, h, w) fused depths of V posed views ->
    (#consistent views, #views that see the pixel, consensus depth).

    The consensus depth is the geometric mean of the pixel's own depth and the depths its
    consistent neighbours imply along the same ray (z_i * d_j / z_ij), as in COLMAP's fusion;
    it pulls the per-view surfaces, which agree only within `tol`, onto one surface."""
    V, h, w = D.shape
    dev = D.device
    v, u = torch.meshgrid(torch.arange(h, device=dev) + 0.5, torch.arange(w, device=dev) + 0.5, indexing="ij")
    Kt = torch.tensor(Kn, dtype=torch.float32, device=dev)
    rays = torch.linalg.solve(Kt, torch.stack([u, v, torch.ones_like(u)]).reshape(3, -1))
    Rt = torch.tensor(R, dtype=torch.float32, device=dev)
    tt = torch.tensor(t, dtype=torch.float32, device=dev)
    n_cons = torch.zeros(V, h * w, device=dev)
    n_vis = torch.zeros(V, h * w, device=dev)
    consensus = torch.zeros(V, h * w, device=dev)
    for i in range(V):
        zi = D[i].reshape(-1)
        Xw = Rt[i].T @ (rays * zi[None] - tt[i][:, None])
        log_sum = torch.log(zi.clamp_min(1e-8))
        for j in range(V):
            if j == i:
                continue
            Xj = Rt[j] @ Xw + tt[j][:, None]
            z = Xj[2]
            p = (Kt @ Xj)[:2] / z.clamp_min(1e-8)
            inb = (z > 1e-6) & (p[0] >= 0) & (p[0] < w) & (p[1] >= 0) & (p[1] < h) & (zi > 0)
            dj = sample(D[j], p.T, mode="nearest")[0]
            ok = inb & (dj > 0) & ((z - dj).abs() < tol * dj)
            n_cons[i] += ok.float()
            n_vis[i] += (inb & (dj > 0)).float()
            log_sum += torch.where(ok, torch.log((zi * dj / z.clamp_min(1e-8)).clamp_min(1e-8)), torch.zeros_like(zi))
        consensus[i] = torch.where(zi > 0, torch.exp(log_sum / (1.0 + n_cons[i])), torch.zeros_like(zi))
    return n_cons.reshape(V, h, w), n_vis.reshape(V, h, w), consensus.reshape(V, h, w)
