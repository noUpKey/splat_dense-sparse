"""Multimodal fusion: SIFT points for the wreck + dense points for the water region.

Writes
  fusion/fused.ply           x y z rgb, source (0 = SIFT/wreck, 1 = dense/water), conf
  fusion/priors/<stem>.npz   per training view: dense depth prior (water region only, network grid)
                             and SIFT keypoint depths (wreck) for the geometric losses
  fusion/masks/<stem>.png    wreck mask (full resolution) for every registered view
  fusion/medium_init.json    data-driven initial water parameters (B_inf, beta_B, beta_D)
"""
import os
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from ..common.camera import PinholeCamera
from ..common.geometry import camera_center
from ..common.io import load_json, load_rgb, log, save_json, save_rgb, write_ply
from ..dense.predict import NetGrid
from ..dense.run import load_scene
from .routing import route, sift_support


@dataclass
class FusionConfig:
    wreck_cell_px: float = 24.0
    min_consistent: int = 2
    min_conf: float = 1.2           # network confidence floor (open water sits at 1.0)
    dense_stride: int = 2
    voxel: float = 0.004            # normalised units (median camera-point range = 1)
    sparse_exclusion: float = 2.0   # drop dense points closer than this many voxels to a SIFT point
    max_dense_points: int = 400_000
    outlier_k: int = 8
    outlier_std: float = 2.5


def _bilinear_rgb(img: np.ndarray, uv: np.ndarray, width: int = 4096) -> np.ndarray:
    """Bilinear colours at full-resolution pixel positions uv (+0.5 convention), (n, 2) -> (n, 3).
    The lookup maps are folded into rows of `width`: cv2.remap rejects maps with >= 32767 rows."""
    n = len(uv)
    if n == 0:
        return np.zeros((0, img.shape[2]), img.dtype)
    m = np.zeros((n + (-n) % width, 2), np.float32)
    m[:n] = uv - 0.5
    out = cv2.remap(img, m[:, 0].reshape(-1, width), m[:, 1].reshape(-1, width), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT)
    return out.reshape(-1, img.shape[2])[:n]


def _voxel_merge(xyz, rgb, conf, view, voxel):
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    n = len(cnt)
    mean = lambda a: np.stack([np.bincount(inv, a[:, k], n) for k in range(a.shape[1])], 1) / cnt[:, None]
    c = np.zeros(n)
    np.maximum.at(c, inv, conf)
    nv = np.unique(np.stack([inv, view], 1), axis=0)[:, 0]
    return mean(xyz), mean(rgb.astype(np.float64)), c, np.bincount(nv, minlength=n)


def _fit_beta_b(ranges, colors, binf, n_bins=10, pct=1.0):
    """Sea-thru style: the darkest pixels at each range are pure backscatter B_inf (1 - exp(-beta_B r))."""
    lum = colors @ np.array([0.299, 0.587, 0.114])
    edges = np.quantile(ranges, np.linspace(0, 1, n_bins + 1))
    r_k, b_k = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (ranges >= a) & (ranges <= b)
        if sel.sum() < 50:
            continue
        thr = np.percentile(lum[sel], pct)
        dark = sel & (lum <= thr)
        r_k.append(np.median(ranges[dark]))
        b_k.append(np.median(colors[dark], 0))
    if len(r_k) < 3:
        return None
    r_k, b_k = np.array(r_k), np.array(b_k)
    betas = np.geomspace(0.01, 10.0, 400)
    model = binf[None, None] * (1 - np.exp(-betas[:, None, None] * r_k[None, :, None]))  # (B, K, 3)
    err = ((model - b_k[None]) ** 2).sum(1)
    return betas[np.argmin(err, 0)]


def run_fusion(out_dir: str, cfg: FusionConfig = FusionConfig()) -> dict:
    meta, pts, R, t, train = load_scene(out_dir)
    summ = load_json(os.path.join(out_dir, "dense", "summary.json"))
    grid = NetGrid(**summ["grid"])
    Kn = np.array(summ["K_net"])
    Kinv = np.linalg.inv(Kn)
    pin = PinholeCamera.from_dict(meta["camera"])
    W, H = pin.width, pin.height
    C = camera_center(R, t)
    for d in ("priors", "masks", "viz"):
        os.makedirs(os.path.join(out_dir, "fusion", d), exist_ok=True)

    o_train = pts["train_ok"][pts["obs_pt"]] & train[pts["obs_img"]]
    dense_xyz, dense_rgb, dense_conf, dense_view = [], [], [], []
    open_rgb, far_rgb, frac = [], [], {}
    for i, im in enumerate(meta["images"]):
        if not im["registered"]:
            continue
        stem = os.path.splitext(im["name"])[0]
        o = (pts["obs_img"] == i) & (o_train if train[i] else pts["train_ok"][pts["obs_pt"]])
        wreck_full = sift_support(pts["obs_uv"][o], W, H, cfg.wreck_cell_px)
        cv2.imwrite(os.path.join(out_dir, "fusion", "masks", stem + ".png"), wreck_full.astype(np.uint8) * 255)
        if not train[i]:
            continue
        dn = np.load(os.path.join(out_dir, "dense", stem + ".npz"))
        img = load_rgb(os.path.join(out_dir, im["file"]))
        small = cv2.resize(img, (grid.w, grid.h_pre), interpolation=cv2.INTER_AREA)[int(grid.oy):int(grid.oy) + grid.h]
        wreck, surf, open_w = route(wreck_full, dn["depth"], dn["n_cons"], dn["conf"],
                                    cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), grid, cfg.min_consistent, cfg.min_conf)
        frac[stem] = {"wreck": float(wreck.mean()), "water_surface": float(surf.mean()), "open_water": float(open_w.mean())}

        ys, xs = np.mgrid[0:grid.h:cfg.dense_stride, 0:grid.w:cfg.dense_stride]
        ys, xs = ys.ravel(), xs.ravel()
        uvn = np.stack([xs + 0.5, ys + 0.5], 1)
        uv_full = (uvn + np.array([0.0, grid.oy])) / np.array([grid.sx, grid.sy])
        z = dn["depth"][ys, xs]
        sel = surf[ys, xs]
        Xc = (np.concatenate([uvn[sel], np.ones((sel.sum(), 1))], 1) @ Kinv.T) * z[sel, None]
        dense_xyz.append((Xc - t[i]) @ R[i])
        dense_rgb.append(_bilinear_rgb(img, uv_full[sel]))
        dense_conf.append(dn["conf"][ys, xs][sel])
        dense_view.append(np.full(sel.sum(), i))
        ow = open_w[ys, xs]
        open_rgb.append(_bilinear_rgb(img, uv_full[ow])[::4])
        valid_z = z[sel]
        if len(valid_z):
            far = sel.copy()
            far[sel] = valid_z >= np.percentile(valid_z, 95)
            far_rgb.append(_bilinear_rgb(img, uv_full[far]))

        kp = o & (pts["obs_img"] == i)
        kp_uv = pts["obs_uv"][kp]
        kp_z = (pts["xyz"][pts["obs_pt"][kp]] @ R[i].T + t[i])[:, 2]
        np.savez_compressed(os.path.join(out_dir, "fusion", "priors", stem + ".npz"),
                            dense_depth=np.where(surf, dn["depth"], 0).astype(np.float32),
                            dense_conf=np.where(surf, dn["conf"], 0).astype(np.float32),
                            kp_uv=kp_uv.astype(np.float32), kp_z=kp_z.astype(np.float32))
        small = small.astype(np.float32)
        overlay = np.zeros_like(small)
        overlay[wreck], overlay[surf], overlay[open_w] = (230, 60, 40), (60, 220, 90), (40, 110, 255)
        save_rgb(os.path.join(out_dir, "fusion", "viz", stem + ".jpg"),
                 np.concatenate([small, 0.45 * small + 0.55 * overlay], 1).astype(np.uint8))

    # ---- sparse (wreck) cloud: SIFT points supported by >= 2 training views
    keep = pts["train_ok"]
    rgb_sum = np.zeros((len(pts["xyz"]), 3))
    np.add.at(rgb_sum, pts["obs_pt"][o_train], pts["obs_rgb"][o_train].astype(np.float64))
    n_obs = np.bincount(pts["obs_pt"][o_train], minlength=len(pts["xyz"]))
    sp_xyz = pts["xyz"][keep].astype(np.float64)
    sp_rgb = rgb_sum[keep] / np.maximum(n_obs[keep, None], 1)

    # ---- dense (water) cloud: voxel de-duplication across views, SIFT priority, outlier removal
    d_xyz, d_rgb = np.concatenate(dense_xyz), np.concatenate(dense_rgb)
    d_conf, d_view = np.concatenate(dense_conf), np.concatenate(dense_view)
    n_raw = len(d_xyz)
    voxel = cfg.voxel
    while True:
        m_xyz, m_rgb, m_conf, m_views = _voxel_merge(d_xyz, d_rgb, d_conf, d_view, voxel)
        if len(m_xyz) <= cfg.max_dense_points:
            break
        voxel *= 1.25
    if len(sp_xyz):
        dist, _ = cKDTree(sp_xyz).query(m_xyz, k=1)
        far_from_sift = dist > cfg.sparse_exclusion * voxel
    else:
        far_from_sift = np.ones(len(m_xyz), bool)
    m_xyz, m_rgb, m_conf, m_views = (a[far_from_sift] for a in (m_xyz, m_rgb, m_conf, m_views))
    if len(m_xyz) > cfg.outlier_k:
        dk, _ = cKDTree(m_xyz).query(m_xyz, k=cfg.outlier_k + 1)
        md = dk[:, 1:].mean(1)
        inl = md < md.mean() + cfg.outlier_std * md.std()
        m_xyz, m_rgb, m_conf, m_views = (a[inl] for a in (m_xyz, m_rgb, m_conf, m_views))

    xyz = np.concatenate([sp_xyz, m_xyz])
    rgb = np.concatenate([sp_rgb, m_rgb])
    source = np.concatenate([np.zeros(len(sp_xyz), np.uint8), np.ones(len(m_xyz), np.uint8)])
    conf = np.concatenate([np.full(len(sp_xyz), -1.0), m_conf]).astype(np.float32)
    write_ply(os.path.join(out_dir, "fusion", "fused.ply"), xyz, np.clip(rgb, 0, 255).round().astype(np.uint8),
              source=source, conf=conf)

    # ---- medium initialisation
    open_rgb = np.concatenate(open_rgb) / 255.0 if open_rgb else np.zeros((0, 3))
    open_frac = np.mean([f["open_water"] for f in frac.values()]) if frac else 0.0
    if open_frac >= 0.005 and len(open_rgb) >= 100:
        binf, src = np.median(open_rgb, 0), "open_water"
    elif far_rgb:
        binf, src = np.median(np.concatenate(far_rgb) / 255.0, 0), "farthest_dense"
    else:
        binf, src = np.full(3, 0.3), "default"
    rng = np.linalg.norm(pts["xyz"][pts["obs_pt"][o_train]] - C[pts["obs_img"][o_train]], axis=1)
    beta_b = _fit_beta_b(rng, pts["obs_rgb"][o_train] / 255.0, binf)
    beta_b = np.full(3, 0.4) if beta_b is None else beta_b
    medium = {"binf": binf.tolist(), "beta_b": np.clip(beta_b, 0.05, 5.0).tolist(),
              "beta_d": np.clip(beta_b, 0.05, 5.0).tolist(), "binf_source": src}
    save_json(os.path.join(out_dir, "fusion", "medium_init.json"), medium)

    stats = {"config": asdict(cfg), "sparse_points": int(len(sp_xyz)), "dense_points": int(len(m_xyz)),
             "dense_raw_samples": int(n_raw), "voxel": voxel, "routing": frac, "medium_init": medium,
             "mean_routing": {k: float(np.mean([f[k] for f in frac.values()])) for k in ("wreck", "water_surface", "open_water")}}
    save_json(os.path.join(out_dir, "fusion", "stats.json"), stats)
    mr = stats["mean_routing"]
    log(f"fusion: {len(sp_xyz)} SIFT (wreck) + {len(m_xyz)} dense (water) points; pixels: wreck {100 * mr['wreck']:.0f}% "
        f"/ water surface {100 * mr['water_surface']:.0f}% / open water {100 * mr['open_water']:.0f}%; "
        f"B_inf={np.round(binf, 3).tolist()} ({src}) beta_B={np.round(beta_b, 3).tolist()}")
    return stats
