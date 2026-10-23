"""Dense stage driver: pointmaps -> SfM-scaled, fused, consistency-checked depth per training view."""
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..common.camera import PinholeCamera
from ..common.io import load_json, load_rgb, log, save_json, save_rgb
from .align import cross_view_consistency, fit_scale, fuse_view, sample
from .predict import DensePredictor, make_grid, select_pairs
from .weights import download

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_weights(path: str) -> str:
    """Accept paths relative to the working directory or the repository; download if missing."""
    if os.path.exists(path):
        return path
    alt = os.path.join(REPO_ROOT, path)
    return alt if os.path.exists(alt) else download(path if os.path.isabs(path) else alt)


@dataclass
class DenseConfig:
    weights: str = "weights/DUSt3R_ViTLarge_BaseDecoder_512_dpt.safetensors"
    width: int = 512
    k_pairs: int = 4
    color_norm: str = "none"         # "none" | "stretch" (per-channel percentile stretch)
    max_pair_rel_err: float = 0.25   # drop pair predictions that disagree with the SfM this much
    pair_agree_tol: float = 0.05
    consistency_tol: float = 0.05
    consensus_iters: int = 2
    batch: int = 4
    device: str = "cuda"
    dtype: str = "bf16"


def load_scene(out_dir: str):
    meta = load_json(os.path.join(out_dir, "sfm", "scene.json"))
    pts = dict(np.load(os.path.join(out_dir, "sfm", "points.npz")))
    R = np.array([im["R"] for im in meta["images"]])
    t = np.array([im["t"] for im in meta["images"]])
    train = np.array([im["registered"] and im["split"] == "train" for im in meta["images"]])
    # strict split: a point may be used only if two training views observe it
    n_train = np.bincount(pts["obs_pt"][train[pts["obs_img"]]], minlength=len(pts["xyz"]))
    pts["train_ok"] = n_train >= 2
    return meta, pts, R, t, train


def run_dense(out_dir: str, cfg: DenseConfig = DenseConfig()) -> dict:
    meta, pts, R, t, train = load_scene(out_dir)
    pin = PinholeCamera.from_dict(meta["camera"])
    views = list(np.where(train)[0])
    images = {i: load_rgb(os.path.join(out_dir, meta["images"][i]["file"])) for i in views}
    grid = make_grid(pin.width, pin.height, pin.pixel_aspect, cfg.width)
    Kn = grid.K_net(pin.K)
    log(f"dense: {len(views)} training views, network input {grid.w}x{grid.h} (square pixels)")

    o_ok = pts["train_ok"][pts["obs_pt"]] & train[pts["obs_img"]]
    pairs = select_pairs(pts["obs_img"][o_ok], pts["obs_pt"][o_ok], views, cfg.k_pairs)
    predictor = DensePredictor(resolve_weights(cfg.weights), cfg.device, cfg.dtype)
    P, C = predictor(images, pairs, grid, cfg.color_norm, cfg.batch)
    depth = P[..., 2]
    del P

    pair_stats, scales = [], np.full(len(pairs), np.nan)
    for p, (i, j) in enumerate(pairs):
        o = o_ok & (pts["obs_img"] == i)
        uv = torch.from_numpy(grid.to_net(pts["obs_uv"][o].astype(np.float64)))
        z = (pts["xyz"][pts["obs_pt"][o]] @ R[i].T + t[i])[:, 2]
        d = sample(depth[p], uv)[0].numpy()
        w = sample(C[p], uv)[0].numpy()
        s, st = fit_scale(d, z, w)
        good = np.isfinite(s) and st.get("median_rel_err", 1.0) < cfg.max_pair_rel_err
        scales[p] = s if good else np.nan
        pair_stats.append({"ref": meta["images"][i]["name"], "src": meta["images"][j]["name"], "scale": s,
                           "used": bool(good), **st})

    fused, fconf, nagree = {}, {}, {}
    for i in views:
        idx = [p for p, (a, _) in enumerate(pairs) if a == i and np.isfinite(scales[p])]
        if not idx:
            fused[i] = torch.zeros(grid.h, grid.w)
            fconf[i], nagree[i] = torch.zeros_like(fused[i]), torch.zeros_like(fused[i])
            continue
        sc = torch.tensor(scales[idx], dtype=torch.float32)[:, None, None]
        fused[i], fconf[i], nagree[i] = fuse_view(depth[idx] * sc, C[idx], cfg.pair_agree_tol)

    dev = cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
    D = torch.stack([fused[i] for i in views]).to(dev)
    # Snap multi-view-supported pixels onto the consensus surface (on Isro two passes halve the
    # inter-view surface dispersion, 1.23% -> 0.66% of depth, and reduce the SfM error 6.8% -> 6.3%).
    for _ in range(cfg.consensus_iters):
        n_cons, _, D_cons = cross_view_consistency(D, Kn, R[views], t[views], cfg.consistency_tol)
        D = torch.where(n_cons > 0, D_cons, D)
    n_cons, n_vis, _ = cross_view_consistency(D, Kn, R[views], t[views], cfg.consistency_tol)
    n_cons, n_vis, D = n_cons.cpu(), n_vis.cpu(), D.cpu()
    for k, i in enumerate(views):
        fused[i] = D[k]

    os.makedirs(os.path.join(out_dir, "dense", "viz"), exist_ok=True)
    per_view = {}
    for k, i in enumerate(views):
        stem = os.path.splitext(meta["images"][i]["name"])[0]
        np.savez_compressed(os.path.join(out_dir, "dense", f"{stem}.npz"), depth=fused[i].numpy(),
                            conf=fconf[i].numpy(), n_agree=nagree[i].numpy().astype(np.uint8),
                            n_cons=n_cons[k].numpy().astype(np.uint8), n_vis=n_vis[k].numpy().astype(np.uint8))
        per_view[stem] = {"consistent_frac_2": float((n_cons[k] >= 2).float().mean()),
                          "mean_conf": float(fconf[i].mean())}
        save_rgb(os.path.join(out_dir, "dense", "viz", f"{stem}.jpg"),
                 _viz(images[i], fused[i].numpy(), n_cons[k].numpy(), fconf[i].numpy(), grid))
    rel = [s["median_rel_err"] for s in pair_stats if "median_rel_err" in s]
    summary = {"config": asdict(cfg), "grid": asdict(grid), "K_net": Kn.tolist(), "views": [int(v) for v in views],
               "pairs": pair_stats, "per_view": per_view,
               "median_pair_rel_err": float(np.median(rel)) if rel else None,
               "pairs_used": int(np.isfinite(scales).sum()), "pairs_total": len(pairs)}
    save_json(os.path.join(out_dir, "dense", "summary.json"), summary)
    log(f"dense: {summary['pairs_used']}/{len(pairs)} pairs used, median SfM agreement error "
        f"{100 * summary['median_pair_rel_err']:.1f}%")
    return summary


def _viz(img, depth, n_cons, conf, grid, min_conf=1.2):
    """Image | log-depth | consistent (>= 2 views) and confident mask, at network resolution."""
    from PIL import Image
    import matplotlib.cm as cm
    im = np.asarray(Image.fromarray(img).resize((grid.w, grid.h_pre)))[int(grid.oy):int(grid.oy) + grid.h]
    ld = np.log(np.where(depth > 0, depth, np.nan))
    lo, hi = np.nanpercentile(ld, [2, 98]) if np.isfinite(ld).any() else (0, 1)
    dc = (cm.turbo(np.nan_to_num((ld - lo) / max(hi - lo, 1e-6), nan=0.0).clip(0, 1))[..., :3] * 255).astype(np.uint8)
    m = ((n_cons >= 2) & (conf >= min_conf))[..., None]
    mask = (im * 0.35 + m * np.array([255, 200, 0]) * 0.65).astype(np.uint8)
    return np.concatenate([im, dc, mask], 1)
