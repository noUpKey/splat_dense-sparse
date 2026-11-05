"""E2: which per-frame camera colour model predicts held-out frames best? Refit on a frozen model; no training.

For every training frame k the frozen model's raw render R_k (no colour correction) is mapped to the photo I_k by one
of these families, fitted by least squares on that frame:

    A    I = W_k R + b_k            3x3 + offset per frame (12)   uwsplat's Appearance module has this form
    B    I = W_k R                  3x3 per frame (9)
    C    I = M diag(g_k) R          shared 3x3 M, per-channel gains (3 per frame); alternating least squares
    D    I = diag(g_k) R            per-channel gains (3)
    Do   I = diag(g_k) R + o_k      gains + offset (6)
    E    I = e_k R                  one exposure scale (1)

A held-out frame gets its parameters from its two temporal training neighbours (heldout.temporal_neighbors; gains are
interpolated in log space, matrices and offsets linearly). Training frames are scored leave-one-out the same way.
Reference rows: identity, and the trained Appearance correction interpolated the same way ("trained", = the
`neighbors` protocol of heldout_appearance.py).

Why (scripts/identifiability_check.py): per-frame offsets are confounded with backscatter (exactly when the medium's
attenuation and backscatter coefficients coincide), so they should absorb water residuals that do not interpolate in
time; a gauge-minimal camera (C, D) should predict held-out frames better. Pre-registered kill rule ("Novel
contributions beyond RUSplatting", E2): C or D must beat A on test PSNR (neighbour interpolation) in >= 3 of 4 scenes;
beat = by more than 0.05 dB (margin fixed before any real run).
"""
import os

import numpy as np
import torch

from ..common.io import log, save_json
from ..gs.data import SceneData
from ..gs.render import render
from .evaluate import load_model
from .heldout import _scores, apply_affine, centred_appearance, temporal_neighbors

FAMILIES = ("A", "B", "C", "D", "Do", "E")


# ------------------------------------------------------------------------------------------------------- fitting
def _lstsq(X, Y, ridge=1e-9):
    X, Y = X.double(), Y.double()
    return torch.linalg.solve(X.T @ X + ridge * torch.eye(X.shape[1], dtype=X.dtype, device=X.device), X.T @ Y)


def fit_frame(fam, X, Y):
    """X, Y (n, 3) raw render / photo pixels of one frame -> parameter dict (C is fitted jointly, see fit_C)."""
    if fam == "A":
        W = _lstsq(torch.cat([X, torch.ones_like(X[:, :1])], 1), Y)            # (4, 3)
        return {"W": W[:3].T, "b": W[3]}
    if fam == "B":
        return {"W": _lstsq(X, Y).T, "b": torch.zeros(3, dtype=torch.float64, device=X.device)}
    if fam == "D":
        g = (X.double() * Y.double()).sum(0) / (X.double() ** 2).sum(0).clamp_min(1e-12)
        return {"g": g.clamp_min(1e-6), "o": torch.zeros(3, dtype=torch.float64, device=X.device)}
    if fam == "Do":
        g, o = [], []
        for c in range(3):
            w = _lstsq(torch.stack([X[:, c], torch.ones_like(X[:, c])], 1), Y[:, c:c + 1])
            g.append(w[0, 0]); o.append(w[1, 0])
        return {"g": torch.stack(g).clamp_min(1e-6), "o": torch.stack(o)}
    if fam == "E":
        e = (X.double() * Y.double()).sum() / (X.double() ** 2).sum().clamp_min(1e-12)
        return {"g": e.clamp_min(1e-6).repeat(3), "o": torch.zeros(3, dtype=torch.float64, device=X.device)}
    raise ValueError(fam)


def fit_C(Xs, Ys, iters=40):
    """Shared M (3x3) and per-frame gains g_k (3) minimising sum_k |X_k diag(g_k) M^T - Y_k|^2 (alternating LS);
    gauge: geometric mean of each channel's gains = 1 (M absorbs the scale)."""
    dev = Xs[0].device
    Xs = [x.double() for x in Xs]
    Ys = [y.double() for y in Ys]
    M = torch.eye(3, dtype=torch.float64, device=dev)
    XtX = [x.T @ x for x in Xs]
    for _ in range(iters):
        G = M.T @ M
        gs = []
        for x, y, xtx in zip(Xs, Ys, XtX):          # y_i = M diag(x_i) g  ->  (G * X^T X) g = sum_i x_i * (M^T y_i)
            gs.append(torch.linalg.solve(G * xtx + 1e-12 * torch.eye(3, dtype=G.dtype, device=dev),
                                         ((y @ M) * x).sum(0)))
        gs = torch.stack(gs).clamp_min(1e-6)
        Z = torch.cat([x * g for x, g in zip(Xs, gs)])
        M = _lstsq(Z, torch.cat(Ys)).T
        s = torch.exp(torch.log(gs).mean(0))       # gauge
        gs, M = gs / s, M * s
    return M, gs


# ------------------------------------------------------------------------------------------------- prediction
def interpolate(fam, params, weights):
    """Blend the training frames' parameters with temporal weights [(p, w), ...]; None -> identity-like."""
    if not weights:
        return None
    if fam in ("A", "B"):
        return {"W": sum(w * params[p]["W"] for p, w in weights), "b": sum(w * params[p]["b"] for p, w in weights)}
    out = {"g": torch.exp(sum(w * torch.log(params[p]["g"]) for p, w in weights)),
           "o": sum(w * params[p]["o"] for p, w in weights)}
    if fam == "C":
        out["M"] = params[weights[0][0]]["M"]
    return out


def apply(fam, img, prm):
    if prm is None:
        return img
    x = img.reshape(-1, 3).double()
    if fam in ("A", "B"):
        y = x @ prm["W"].T + prm["b"]
    elif fam == "C":
        y = (x * prm["g"]) @ prm["M"].T
    else:
        y = x * prm["g"] + prm["o"]
    return y.float().reshape(img.shape)


# ------------------------------------------------------------------------------------------------- experiment
@torch.no_grad()
def run(out_dir, tag, device="cuda", backend=None, families=FAMILIES, stride=2, lpips_net="vgg"):
    ckpt_path = os.path.join(out_dir, "gs", tag, "ckpt.pt")
    params, medium, bg, cfg = load_model(ckpt_path, device)
    app_state = torch.load(ckpt_path, map_location=device, weights_only=False).get("appearance")
    backend = backend or cfg["backend"]
    data = SceneData(out_dir, device, cfg["downscale"], load_priors=False)
    train = sorted(data.train, key=lambda v: v.train_id)
    train_idx = [v.idx for v in train]
    dA, db = centred_appearance(app_state, len(train), device)
    sh = cfg["sh_degree"]

    raw = [render(params, v, medium, sh, backend, bg_color=bg)["image"] for v in train]
    Xs = [r[::stride, ::stride].reshape(-1, 3) for r in raw]                 # fit on a pixel subsample
    Ys = [v.image[::stride, ::stride].reshape(-1, 3) for v in train]
    fits = {}
    for fam in families:
        if fam == "C":
            M, gs = fit_C(Xs, Ys)
            fits[fam] = [{"M": M, "g": g, "o": torch.zeros(3, dtype=torch.float64, device=device)} for g in gs]
        else:
            fits[fam] = [fit_frame(fam, x, y) for x, y in zip(Xs, Ys)]

    def score(img, gt, full=True):
        return _scores(img, gt, None, lpips_net, full=full)

    # training views, leave-one-out (neighbours without the frame itself) and own fit
    loo = {k: [] for k in ("identity", "trained") + tuple(families)}
    own = {k: [] for k in families}
    for p, v in enumerate(train):
        w = temporal_neighbors(v.idx, train_idx, exclude=p)
        loo["identity"].append(score(raw[p], v.image, False)["psnr"])
        At = sum(wt * dA[q] for q, wt in w) if w else torch.zeros(3, 3, device=device)
        bt = sum(wt * db[q] for q, wt in w) if w else torch.zeros(3, device=device)
        loo["trained"].append(score(apply_affine(raw[p], At, bt), v.image, False)["psnr"])
        for fam in families:
            loo[fam].append(score(apply(fam, raw[p], interpolate(fam, fits[fam], w)), v.image, False)["psnr"])
            own[fam].append(score(apply(fam, raw[p], fits[fam][p]), v.image, False)["psnr"])

    test = {}
    for v in data.test:
        img = render(params, v, medium, sh, backend, bg_color=bg)["image"]
        w = temporal_neighbors(v.idx, train_idx)
        rec = {"frame": v.idx, "neighbours": [[train[q].stem, round(float(wt), 3)] for q, wt in w]}
        rec["identity"] = score(img, v.image)
        At = sum(wt * dA[q] for q, wt in w) if w else torch.zeros(3, 3, device=device)
        bt = sum(wt * db[q] for q, wt in w) if w else torch.zeros(3, device=device)
        rec["trained"] = score(apply_affine(img, At, bt), v.image)
        for fam in families:
            rec[fam] = score(apply(fam, img, interpolate(fam, fits[fam], w)), v.image)
        test[v.name] = rec
        log(f"  {v.name}: " + " | ".join(f"{k} {rec[k]['psnr']:.2f}" for k in ("identity", "trained") + tuple(families)))

    keys = ("identity", "trained") + tuple(families)
    summary = {"test": {k: {m: float(np.mean([r[k][m] for r in test.values()])) for m in ("psnr", "ssim", "lpips")}
                        for k in keys},
               "train_loo_psnr": {k: float(np.mean(v)) for k, v in loo.items()},
               "train_own_psnr": {k: float(np.mean(v)) for k, v in own.items()}}
    # offsets vs backscatter (E3-lite): per-frame offset of family Do next to the frame's mean backscatter share
    extra = {}
    if "Do" in fits:
        extra["Do_offsets"] = [[float(x) for x in f["o"]] for f in fits["Do"]]
        extra["Do_gains"] = [[float(x) for x in f["g"]] for f in fits["Do"]]
    if "C" in fits:
        extra["C_M"] = fits["C"][0]["M"].tolist()
    res = {"scene": data.meta["scene"], "tag": tag, "stride": stride, "summary": summary, "test": test,
           "train_loo": {train[p].name: {k: loo[k][p] for k in loo} for p in range(len(train))}, "fits": extra}
    save_json(os.path.join(out_dir, "eval", tag, "camera_families.json"), res)
    log(f"[camera families] {data.meta['scene']} test PSNR: " +
        " | ".join(f"{k} {summary['test'][k]['psnr']:.2f}" for k in keys))
    log(f"[camera families] {data.meta['scene']} train leave-one-out PSNR: " +
        " | ".join(f"{k} {summary['train_loo_psnr'][k]:.2f}" for k in keys))
    return res
