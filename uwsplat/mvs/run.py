"""MVS stage: photometrically verified dense geometry ("dense cloud v2").

Why (measured, Oct 2026): extra *triangulated* points helped (+0.85 / +0.89 dB on Cormoran / Tokai from
zero-weight interpolated frames that only added COLMAP tracks), network geometry on its own did not
(VGGT -> 3DGS 19.4 dB; DUSt3R depth is ~6 % off the SfM depths), and fusion v1 therefore kept dense
points off the wreck, so the wreck - where the texture and the SSIM / LPIPS-relevant detail are -
started from sparse SIFT points only. Here the dense network is only the *prior*: every textured pixel
is re-measured by multi-view NCC with the SfM poses, so dense points reach triangulation-grade accuracy
and can be placed on the wreck too.

Per training view (training views only, leakage-free):
  1. prior pass: hypotheses around the dense network's depth (dense/<stem>.npz, if present);
  2. wide pass over the view's SIFT depth range for textured pixels the prior pass did not verify
     (the only pass when there is no dense stage);
     hypotheses are spaced so that the widest-baseline source moves <= px_per_step grid pixels per step;
  3. texture gate + aggregated NCC -> photometrically verified depth; multi-view geometric consistency
     between the verified depth maps of different views, with consensus snapping (COLMAP-fusion style).
Pixel classes:
  verified  (2): photometrically + geometrically verified surface, wreck or seabed -> dense points
  completed (1): not verified, but v1 routing calls it a surface (confident, consistent network depth,
                 outside the wreck) -> network points exactly as fusion v1
  open water(0): neither -> rendered by the medium
Outputs (mvs/): fused.ply (source 0 SIFT / 1 completed / 2 verified, conf, support, unit normal nx ny nz;
0 where unknown), priors/<stem>.npz (depth prior on the MVS grid + SIFT keypoint depths, same keys as
fusion/priors, plus cls), masks/<stem>.png (SIFT-support wreck masks, as fusion), medium_init.json (copied
from fusion when it exists, so v1 and v2 start from the same water), viz/<stem>.jpg, summary.json (grid,
per-view fractions, agreement with SIFT depths that the sweep never uses).
"""
import os
import shutil
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

from ..common.camera import PinholeCamera
from ..common.geometry import camera_center
from ..common.io import load_json, load_rgb, log, save_json, save_rgb, write_ply
from ..dense.align import cross_view_consistency
from ..dense.predict import NetGrid
from ..dense.run import load_scene
from ..fusion.fuse import FusionConfig, _bilinear_rgb, _fit_beta_b
from ..fusion.routing import route, sift_support
from .sweep import best_depth, ncc_volume, window_std

SIFT, COMPLETED, VERIFIED = 0, 1, 2


@dataclass
class MVSConfig:
    scale: float = 0.5              # MVS grid = scale x undistorted frame (1280x720 -> 640x360)
    n_src: int = 4                  # source views per reference view
    use_aux: bool = True            # interpolated 'aux' frames (if any) may serve as source views
    min_angle: float = 2.0          # deg; median triangulation angle of shared SIFT points
    good_angle: float = 6.0
    max_angle: float = 35.0
    window: int = 7                 # NCC window (MVS-grid pixels)
    prior_range: float = 0.2        # prior pass: +-0.2 in log depth (~ +-22 %) around the network depth
    wide_margin: float = 0.35       # wide pass: log margin around the 2-98 % SIFT depth percentiles
    px_per_step: float = 0.75       # hypothesis spacing, in grid pixels of the widest-baseline source
    n_hyp_min: int = 32
    n_hyp_max_prior: int = 96
    n_hyp_max_wide: int = 192
    topk: int = 2                   # aggregate the k best sources per hypothesis
    min_ncc: float = 0.5
    min_std: float = 2.0            # texture gate: std of the reference window (grey levels)
    min_valid: float = 0.9
    consistency_tol: float = 0.015  # relative depth agreement between views
    min_consistent: int = 1         # other views whose verified depth agrees
    consensus_iters: int = 1
    use_prior: bool = True          # use dense/<stem>.npz when present
    complete: bool = True           # add v1-style network points where nothing was verified
    verified_stride: int = 1
    completed_stride: int = 2       # on the network grid, as fusion v1
    voxel: float = 0.003            # normalised units (median camera-point range = 1)
    max_verified_points: int = 800_000
    max_completed_points: int = 200_000
    sparse_exclusion: float = 1.5   # drop dense points closer than this many voxels to a SIFT point
    outlier_k: int = 8
    outlier_std: float = 2.5
    normal_step: int = 2            # normals from central differences over +-normal_step grid pixels
    normal_max_jump: float = 0.03   # ... only where the relative depth change over that span is below this
    w_verified: float = 1.0         # depth-prior weights used by the trainer
    w_completed: float = 0.3
    chunk: int = 16
    device: str = "cuda"


def mvs_grid(W: int, H: int, scale: float) -> NetGrid:
    w, h = int(round(W * scale)), int(round(H * scale))
    return NetGrid(W, H, w, h, h, w / W, h / H, 0.0)


def to_grid(img: np.ndarray, g: NetGrid) -> np.ndarray:
    """Full-resolution image -> grid g (area resampling, then the crop of g)."""
    small = cv2.resize(img, (g.w, g.h_pre), interpolation=cv2.INTER_AREA)
    return small[int(g.oy):int(g.oy) + g.h]


def resample(m: np.ndarray, src: NetGrid, dst: NetGrid) -> np.ndarray:
    """Nearest resampling of a map on grid `src` onto grid `dst` (zeros outside src)."""
    u = (np.arange(dst.w) + 0.5) / dst.sx * src.sx
    v = ((np.arange(dst.h) + 0.5) + dst.oy) / dst.sy * src.sy - src.oy
    iu, iv = np.floor(u).astype(int), np.floor(v).astype(int)
    out = np.zeros((dst.h, dst.w), m.dtype)
    okv, oku = (iv >= 0) & (iv < src.h), (iu >= 0) & (iu < src.w)
    out[np.ix_(okv, oku)] = m[np.ix_(iv[okv], iu[oku])]
    return out


def select_sources(pts, C, views, o_ok, cfg: MVSConfig, candidates=None) -> dict:
    """For each view: the n_src views (from `candidates`, default the views themselves) sharing the most SIFT
    points at a useful triangulation angle."""
    candidates = list(views) if candidates is None else list(candidates)
    vis = {v: np.unique(pts["obs_pt"][o_ok & (pts["obs_img"] == v)]) for v in dict.fromkeys(list(views) + candidates)}
    out = {}
    for i in views:
        cand = []
        for j in candidates:
            if j == i:
                continue
            shared = np.intersect1d(vis[i], vis[j], assume_unique=True)
            if len(shared) < 10:
                continue
            X = pts["xyz"][shared].astype(np.float64)
            a, b = C[i] - X, C[j] - X
            cos = np.sum(a * b, 1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
            ang = float(np.degrees(np.median(np.arccos(np.clip(cos, -1, 1)))))
            w = min(1.0, ang / cfg.good_angle)
            if ang < cfg.min_angle or ang > cfg.max_angle:
                w *= 0.1  # tiny baseline (depth barely constrained) or very different viewpoint
            cand.append((len(shared) * w, int(j), ang, int(len(shared))))
        cand.sort(reverse=True)
        out[i] = cand[:cfg.n_src]
    return out


def n_hypotheses(half: float, f: float, baseline_over_depth: float, cfg: MVSConfig, n_max: int) -> int:
    """Steps over [-half, half] in log depth so the widest-baseline source moves <= px_per_step per step
    (a point at depth z shifts by ~ f * b / z grid pixels per unit of log depth)."""
    n = int(np.ceil(2.0 * half * f * baseline_over_depth / cfg.px_per_step)) + 1
    return int(np.clip(n, cfg.n_hyp_min, n_max))


def relative_pose(R, t, i, j) -> np.ndarray:
    """[R | t] mapping camera-i coordinates to camera-j coordinates."""
    Rr = R[j] @ R[i].T
    return np.concatenate([Rr, (t[j] - Rr @ t[i])[:, None]], 1)


def backproject(uv: np.ndarray, z: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    Xc = (np.concatenate([uv, np.ones((len(uv), 1))], 1) @ np.linalg.inv(K).T) * z[:, None]
    return (Xc - t) @ R


def depth_normals(D: np.ndarray, K: np.ndarray, step: int = 2, max_jump: float = 0.03) -> np.ndarray:
    """Camera-frame unit normals (h, w, 3) of a depth map (0 = invalid) from central differences over
    +-step pixels, facing the camera; zero where a neighbour is missing or the surface jumps."""
    h, w = D.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    rays = np.stack([u + 0.5, v + 0.5, np.ones_like(u)], -1) @ np.linalg.inv(K).T
    X = rays * D[..., None]
    nrm = np.zeros((h, w, 3))
    s = step
    c = (slice(s, h - s), slice(s, w - s))
    l, r = (slice(s, h - s), slice(0, w - 2 * s)), (slice(s, h - s), slice(2 * s, w))
    up, dn = (slice(0, h - 2 * s), slice(s, w - s)), (slice(2 * s, h), slice(s, w - s))
    d0 = D[c]
    ok = (d0 > 0) & (D[l] > 0) & (D[r] > 0) & (D[up] > 0) & (D[dn] > 0)
    for a, b in ((l, r), (up, dn)):
        ok &= np.abs(D[b] - D[a]) < max_jump * np.maximum(d0, 1e-9)
    n = np.cross(X[r] - X[l], X[dn] - X[up])
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    n *= -np.sign(np.sum(n * X[c], -1, keepdims=True) + 1e-12)  # face the camera
    nrm[c] = np.where(ok[..., None], n, 0.0)
    return nrm


def _voxel_merge(xyz, rgb, conf, view, nrm, voxel):
    """Mean position / colour, max confidence, #distinct views and summed (re-normalised) normals per voxel."""
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    n = len(cnt)
    tot = lambda a: np.stack([np.bincount(inv, a[:, k], n) for k in range(a.shape[1])], 1)
    c = np.full(n, -np.inf)
    np.maximum.at(c, inv, conf)
    nv = np.unique(np.stack([inv, view], 1), axis=0)[:, 0]
    ns = tot(nrm)
    nn = np.linalg.norm(ns, axis=1, keepdims=True)
    ns = np.where(nn > 1e-6, ns / np.maximum(nn, 1e-12), 0.0)
    return tot(xyz) / cnt[:, None], tot(rgb.astype(np.float64)) / cnt[:, None], c, np.bincount(nv, minlength=n), ns


def _merge(xyz, rgb, conf, view, nrm, voxel, cap):
    while True:
        m = _voxel_merge(xyz, rgb, conf, view, nrm, voxel)
        if len(m[0]) <= cap:
            return m, voxel
        voxel *= 1.25


def _keep_far(xyz, others, dist):
    if len(others) == 0 or len(xyz) == 0:
        return np.ones(len(xyz), bool)
    d, _ = cKDTree(others).query(xyz, k=1)
    return d > dist


def _inliers(xyz, k, n_std):
    if len(xyz) <= k:
        return np.ones(len(xyz), bool)
    dk, _ = cKDTree(xyz).query(xyz, k=k + 1)
    md = dk[:, 1:].mean(1)
    return md < md.mean() + n_std * md.std()


def _viz(img_small, depth, cls):
    import matplotlib.cm as cm
    ld = np.log(np.where(depth > 0, depth, np.nan))
    lo, hi = np.nanpercentile(ld, [2, 98]) if np.isfinite(ld).any() else (0.0, 1.0)
    dc = (cm.turbo(np.nan_to_num((ld - lo) / max(hi - lo, 1e-6), nan=0.0).clip(0, 1))[..., :3] * 255)
    dc[depth <= 0] = 0
    over = np.zeros_like(img_small, dtype=np.float32)
    over[cls == VERIFIED], over[cls == COMPLETED], over[cls == 0] = (230, 60, 40), (60, 220, 90), (40, 110, 255)
    blend = 0.45 * img_small + 0.55 * over
    return np.concatenate([img_small, dc.astype(np.uint8), blend.astype(np.uint8)], 1)


def _agreement(rel):
    rel = np.asarray(rel)
    if len(rel) == 0:
        return {"n": 0}
    return {"n": int(len(rel)), "median_rel_err": float(np.median(rel)), "p90_rel_err": float(np.percentile(rel, 90)),
            "frac_within_1pct": float(np.mean(rel < 0.01)), "frac_within_5pct": float(np.mean(rel < 0.05))}


def _medium_init(out_dir, meta, pts, C, o_ok, open_rgb, far_rgb, open_frac):
    """Same water start as v1 when fusion ran (so v1 / v2 differ only in geometry); else fusion's recipe on
    the v2 classes: B_inf = median open-water colour (or the farthest verified points), beta_B Sea-thru fit."""
    src = os.path.join(out_dir, "fusion", "medium_init.json")
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(out_dir, "mvs", "medium_init.json"))
        return {**load_json(src), "copied_from": "fusion"}
    open_rgb = np.concatenate(open_rgb) / 255.0 if open_rgb else np.zeros((0, 3))
    if open_frac >= 0.005 and len(open_rgb) >= 100:
        binf, how = np.median(open_rgb, 0), "open_water"
    elif far_rgb:
        binf, how = np.median(np.concatenate(far_rgb) / 255.0, 0), "farthest_verified"
    else:
        binf, how = np.full(3, 0.3), "default"
    rng = np.linalg.norm(pts["xyz"][pts["obs_pt"][o_ok]] - C[pts["obs_img"][o_ok]], axis=1)
    beta_b = _fit_beta_b(rng, pts["obs_rgb"][o_ok] / 255.0, binf)
    beta_b = np.full(3, 0.4) if beta_b is None else beta_b
    medium = {"binf": binf.tolist(), "beta_b": np.clip(beta_b, 0.05, 5.0).tolist(),
              "beta_d": np.clip(beta_b, 0.05, 5.0).tolist(), "binf_source": how}
    save_json(os.path.join(out_dir, "mvs", "medium_init.json"), medium)
    return medium


def run_mvs(out_dir: str, cfg: MVSConfig = MVSConfig()) -> dict:
    t_start = time.time()
    meta, pts, R, t, train = load_scene(out_dir)
    pin = PinholeCamera.from_dict(meta["camera"])
    W, H = pin.width, pin.height
    g = mvs_grid(W, H, cfg.scale)
    Km = g.K_net(pin.K)
    f_grid = float(max(Km[0, 0], Km[1, 1]))
    dev = cfg.device if (torch.cuda.is_available() and cfg.device.startswith("cuda")) else "cpu"
    views = [int(v) for v in np.where(train)[0]]
    # interpolated frames between consecutive training frames (scripts/make_interframes.py): extra source views
    # for the photometric check only; points still need two real training observations (train_ok)
    aux = [i for i, im in enumerate(meta["images"]) if im["registered"] and im["split"] == "aux"] if cfg.use_aux else []
    C = camera_center(R, t)
    o_ok = pts["train_ok"][pts["obs_pt"]] & train[pts["obs_img"]]
    is_src = train.copy()
    is_src[aux] = True
    o_src = pts["train_ok"][pts["obs_pt"]] & is_src[pts["obs_img"]]
    srcs = select_sources(pts, C, views, o_src, cfg, views + aux)
    stems = {i: os.path.splitext(im["name"])[0] for i, im in enumerate(meta["images"])}

    dense_dir = os.path.join(out_dir, "dense")
    have_prior = cfg.use_prior and os.path.exists(os.path.join(dense_dir, "summary.json"))
    ng = NetGrid(**load_json(os.path.join(dense_dir, "summary.json"))["grid"]) if have_prior else None
    fus_stats = os.path.join(out_dir, "fusion", "stats.json")
    fcfg = FusionConfig(**load_json(fus_stats)["config"]) if os.path.exists(fus_stats) else FusionConfig()
    log(f"mvs: {len(views)} training views{f' + {len(aux)} interpolated source frames' if aux else ''} "
        f"on a {g.w}x{g.h} grid, {cfg.n_src} sources each, "
        f"{'network prior + wide pass' if have_prior else 'wide pass only (no dense stage)'}")
    for d in ("priors", "viz", "masks"):
        os.makedirs(os.path.join(out_dir, "mvs", d), exist_ok=True)

    imgs, grays, prior = {}, {}, {}
    for i in views + aux:
        imgs[i] = load_rgb(os.path.join(out_dir, meta["images"][i]["file"]))
        grays[i] = cv2.cvtColor(to_grid(imgs[i], g), cv2.COLOR_RGB2GRAY).astype(np.float32)
        if have_prior and i in views:
            prior[i] = resample(np.load(os.path.join(dense_dir, stems[i] + ".npz"))["depth"].astype(np.float32), ng, g)

    depth, score, textured, n_hyp_used = {}, {}, {}, {}
    for i in views:
        t0 = time.time()
        ref = torch.from_numpy(grays[i]).to(dev)
        textured[i] = (window_std(ref, cfg.window) >= cfg.min_std).cpu().numpy()
        js = [c[1] for c in srcs[i]]
        if not js:
            depth[i], score[i] = np.zeros((g.h, g.w), np.float32), np.full((g.h, g.w), -1.0, np.float32)
            continue
        S = torch.from_numpy(np.stack([grays[j] for j in js])).to(dev)
        rel = torch.tensor(np.stack([relative_pose(R, t, i, j) for j in js]), dtype=torch.float32, device=dev)
        K = torch.tensor(Km, dtype=torch.float32, device=dev)
        sel = o_ok & (pts["obs_img"] == i)
        z = (pts["xyz"][pts["obs_pt"][sel]] @ R[i].T + t[i])[:, 2]
        z = z[z > 0]
        lo, hi = np.percentile(z, [2, 98]) if len(z) >= 20 else (0.3, 3.0)
        z_med = float(np.median(z)) if len(z) else 1.0
        bz = max(float(np.linalg.norm(C[i] - C[j])) for j in js) / max(z_med, 1e-6)
        lo, hi = lo * np.exp(-cfg.wide_margin), hi * np.exp(cfg.wide_margin)
        wc = torch.full((g.h, g.w), float(np.sqrt(lo * hi)), device=dev)
        wh = torch.full_like(wc, float(0.5 * np.log(hi / lo)))
        d_best = torch.zeros_like(wc)
        s_best = torch.full_like(wc, -1.0)
        ok = torch.zeros_like(wc, dtype=torch.bool)
        passes = []
        if have_prior:
            p = torch.from_numpy(prior[i]).to(dev)
            has = p > 0
            passes.append(("prior", torch.where(has, p, wc), torch.where(has, torch.full_like(wh, cfg.prior_range), wh),
                           n_hypotheses(cfg.prior_range, f_grid, bz, cfg, cfg.n_hyp_max_prior)))
        passes.append(("wide", wc, wh, n_hypotheses(float(wh[0, 0]), f_grid, bz, cfg, cfg.n_hyp_max_wide)))
        tex = torch.from_numpy(textured[i]).to(dev)
        n_hyp_used[i] = {}
        for name, d0, half, n_hyp in passes:
            todo = tex & ~ok
            if not bool(todo.any()):
                break
            n_hyp_used[i][name] = n_hyp
            ncc, tau = ncc_volume(ref, S, K, rel, d0, half, n_hyp, cfg.window, cfg.min_valid, cfg.chunk)
            d, s, v = best_depth(ncc, tau, d0, half, cfg.topk, cfg.min_ncc)
            del ncc
            take = todo & v
            d_best = torch.where(take, d, d_best)
            s_best = torch.where(take, s, s_best)
            ok |= take
        depth[i] = torch.where(ok, d_best, torch.zeros_like(d_best)).cpu().numpy()
        score[i] = s_best.cpu().numpy()
        log(f"mvs: {meta['images'][i]['name']}: sources {[meta['images'][j]['name'][-8:-4] for j in js]} "
            f"(angles {[round(c[2], 1) for c in srcs[i]]} deg, b/z {bz:.2f}, hyp {n_hyp_used[i]}), textured "
            f"{100 * textured[i].mean():.0f}%, photometrically verified {100 * (depth[i] > 0).mean():.0f}% "
            f"({time.time() - t0:.1f}s)")

    # ---- multi-view geometric consistency between the verified depth maps, consensus snapping
    Dt = torch.from_numpy(np.stack([depth[i] for i in views])).to(dev)
    for _ in range(cfg.consensus_iters):
        n_cons, _, D_cons = cross_view_consistency(Dt, Km, R[views], t[views], cfg.consistency_tol)
        Dt = torch.where(n_cons > 0, D_cons, Dt)
    n_cons, _, _ = cross_view_consistency(Dt, Km, R[views], t[views], cfg.consistency_tol)
    Dt, n_cons = Dt.cpu().numpy(), n_cons.cpu().numpy()

    # SIFT-support (wreck) masks for every registered view, as fusion v1 writes them (region-wise metrics)
    for i, im in enumerate(meta["images"]):
        if im["registered"]:
            o = (pts["obs_img"] == i) & (o_ok if train[i] else pts["train_ok"][pts["obs_pt"]])
            cv2.imwrite(os.path.join(out_dir, "mvs", "masks", stems[i] + ".png"),
                        sift_support(pts["obs_uv"][o], W, H, fcfg.wreck_cell_px).astype(np.uint8) * 255)

    ver, com = {k: [] for k in ("xyz", "rgb", "conf", "view", "nrm")}, {k: [] for k in ("xyz", "rgb", "conf", "view", "nrm")}
    per_view, agree_mvs, agree_prior, agree_both_mvs, agree_both_prior = {}, [], [], [], []
    open_rgb, far_rgb = [], []
    for k, i in enumerate(views):
        stem = stems[i]
        verified = (Dt[k] > 0) & (n_cons[k] >= cfg.min_consistent)
        dmap = np.where(verified, Dt[k], 0).astype(np.float32)
        cls = np.where(verified, VERIFIED, 0).astype(np.uint8)
        sel = o_ok & (pts["obs_img"] == i)
        kp_uv = pts["obs_uv"][sel].astype(np.float32)
        kp_z = (pts["xyz"][pts["obs_pt"][sel]] @ R[i].T + t[i])[:, 2].astype(np.float32)
        wreck_full = sift_support(kp_uv, W, H, fcfg.wreck_cell_px)

        # completed (network-only) surface: v1 routing, minus what MVS verified
        completed = np.zeros_like(verified)
        if have_prior and cfg.complete:
            dn = np.load(os.path.join(dense_dir, stem + ".npz"))
            gray_net = cv2.cvtColor(to_grid(imgs[i], ng), cv2.COLOR_RGB2GRAY)
            _, surf, _ = route(wreck_full, dn["depth"], dn["n_cons"], dn["conf"], gray_net, ng,
                               fcfg.min_consistent, fcfg.min_conf)
            ver_net = resample(verified.astype(np.uint8), g, ng).astype(bool)
            com_net = surf & ~ver_net
            Kn = ng.K_net(pin.K)
            n_net = depth_normals(np.where(surf, dn["depth"], 0).astype(np.float64), Kn, 1, cfg.normal_max_jump)
            ys, xs = np.mgrid[0:ng.h:cfg.completed_stride, 0:ng.w:cfg.completed_stride]
            ys, xs = ys.ravel(), xs.ravel()
            m = com_net[ys, xs]
            uvn = np.stack([xs[m] + 0.5, ys[m] + 0.5], 1)
            com["xyz"].append(backproject(uvn, dn["depth"][ys[m], xs[m]], Kn, R[i], t[i]))
            com["rgb"].append(_bilinear_rgb(imgs[i], (uvn + np.array([0.0, ng.oy])) / np.array([ng.sx, ng.sy])))
            com["conf"].append(dn["conf"][ys[m], xs[m]])
            com["view"].append(np.full(int(m.sum()), i))
            com["nrm"].append(n_net[ys[m], xs[m]] @ R[i])  # camera -> world (R^T applied to row vectors)
            completed = resample(com_net.astype(np.uint8), ng, g).astype(bool) & ~verified & (prior[i] > 0)
            dmap = np.where(completed, prior[i], dmap).astype(np.float32)
            cls[completed] = COMPLETED

        # verified points (MVS grid) with normals from the consensus depth
        nv = depth_normals(np.where(verified, Dt[k], 0).astype(np.float64), Km, cfg.normal_step, cfg.normal_max_jump)
        ys, xs = np.mgrid[0:g.h:cfg.verified_stride, 0:g.w:cfg.verified_stride]
        ys, xs = ys.ravel(), xs.ravel()
        m = verified[ys, xs]
        uvm = np.stack([xs[m] + 0.5, ys[m] + 0.5], 1)
        ver["xyz"].append(backproject(uvm, Dt[k][ys[m], xs[m]], Km, R[i], t[i]))
        ver["rgb"].append(_bilinear_rgb(imgs[i], uvm / np.array([g.sx, g.sy])))
        ver["conf"].append(score[i][ys[m], xs[m]])
        ver["view"].append(np.full(int(m.sum()), i))
        ver["nrm"].append(nv[ys[m], xs[m]] @ R[i])

        # water samples for the medium start (used only when fusion v1 did not run)
        open_w = (cls == 0) & ~textured[i] & ~resample(wreck_full.astype(np.uint8), mvs_grid(W, H, 1.0), g).astype(bool)
        oy, ox = np.nonzero(open_w)
        open_rgb.append(_bilinear_rgb(imgs[i], np.stack([ox + 0.5, oy + 0.5], 1)[::4] / np.array([g.sx, g.sy])))
        if verified.any():
            far = verified & (Dt[k] >= np.percentile(Dt[k][verified], 95))
            fy, fx = np.nonzero(far)
            far_rgb.append(_bilinear_rgb(imgs[i], np.stack([fx + 0.5, fy + 0.5], 1) / np.array([g.sx, g.sy])))

        weight = np.where(verified, cfg.w_verified, np.where(completed, cfg.w_completed, 0.0)).astype(np.float32)
        np.savez_compressed(os.path.join(out_dir, "mvs", "priors", stem + ".npz"), dense_depth=dmap,
                            dense_conf=weight, kp_uv=kp_uv, kp_z=kp_z, cls=cls)
        save_rgb(os.path.join(out_dir, "mvs", "viz", stem + ".jpg"), _viz(to_grid(imgs[i], g), dmap, cls))

        # agreement with SIFT depths (not used by the sweep beyond picking sources and the wide range)
        uvg = kp_uv * np.array([g.sx, g.sy])
        iu = np.clip(np.floor(uvg[:, 0]).astype(int), 0, g.w - 1)
        iv = np.clip(np.floor(uvg[:, 1]).astype(int), 0, g.h - 1)
        dm = np.where(verified, Dt[k], 0)[iv, iu]
        hit = dm > 0
        agree_mvs.append(np.abs(dm[hit] / kp_z[hit] - 1))
        wreck_g = resample(wreck_full.astype(np.uint8), mvs_grid(W, H, 1.0), g).astype(bool)
        pv = {"verified": float(verified.mean()), "completed": float(completed.mean()),
              "open_water": float(1 - verified.mean() - completed.mean()), "textured": float(textured[i].mean()),
              "verified_on_wreck": float(verified[wreck_g].mean()) if wreck_g.any() else None,
              "verified_off_wreck": float(verified[~wreck_g].mean()) if (~wreck_g).any() else None,
              "keypoint_coverage": float(hit.mean()) if len(hit) else None,
              "n_hyp": n_hyp_used.get(i, {}),
              "sources": [{"name": meta["images"][c[1]]["name"], "angle_deg": round(c[2], 2), "shared": c[3]}
                          for c in srcs[i]]}
        if have_prior:
            dp = prior[i][iv, iu]
            hp = dp > 0
            agree_prior.append(np.abs(dp[hp] / kp_z[hp] - 1))
            both = hit & hp
            agree_both_mvs.append(np.abs(dm[both] / kp_z[both] - 1))
            agree_both_prior.append(np.abs(dp[both] / kp_z[both] - 1))
        per_view[stem] = pv

    # ---- point cloud: SIFT (priority) + verified + completed
    keep = pts["train_ok"]
    rgb_sum = np.zeros((len(pts["xyz"]), 3))
    np.add.at(rgb_sum, pts["obs_pt"][o_ok], pts["obs_rgb"][o_ok].astype(np.float64))
    n_obs = np.bincount(pts["obs_pt"][o_ok], minlength=len(pts["xyz"]))
    sp_xyz = pts["xyz"][keep].astype(np.float64)
    sp_rgb = rgb_sum[keep] / np.maximum(n_obs[keep, None], 1)

    def merged(parts, voxel, cap, others):
        """Voxel de-duplication across views, SIFT (and earlier classes) priority, outlier removal."""
        xyz = np.concatenate(parts["xyz"]) if parts["xyz"] else np.zeros((0, 3))
        if len(xyz) == 0:
            return [np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0, int), np.zeros((0, 3))], voxel
        cat = lambda key: np.concatenate(parts[key])
        out, vox = _merge(xyz, cat("rgb").reshape(-1, 3), cat("conf"), cat("view").astype(int),
                          cat("nrm").reshape(-1, 3), voxel, cap)
        f = _keep_far(out[0], others, cfg.sparse_exclusion * vox)
        out = [a[f] for a in out]
        f = _inliers(out[0], cfg.outlier_k, cfg.outlier_std)
        return [a[f] for a in out], vox

    (vx, vr, vc, vs, vn), v_vox = merged(ver, cfg.voxel, cfg.max_verified_points, sp_xyz)
    (cx, cr, cc, cs, cn), c_vox = merged(com, fcfg.voxel, cfg.max_completed_points, np.concatenate([sp_xyz, vx]))

    xyz = np.concatenate([sp_xyz, cx, vx])
    rgb = np.concatenate([sp_rgb, cr, vr])
    nrm = np.concatenate([np.zeros_like(sp_xyz), cn, vn]).astype(np.float32)
    source = np.concatenate([np.full(len(sp_xyz), SIFT), np.full(len(cx), COMPLETED), np.full(len(vx), VERIFIED)])
    conf = np.concatenate([np.full(len(sp_xyz), -1.0), cc, vc]).astype(np.float32)
    support = np.concatenate([n_obs[keep], cs, vs]).astype(np.int32)
    write_ply(os.path.join(out_dir, "mvs", "fused.ply"), xyz, np.clip(rgb, 0, 255).round().astype(np.uint8),
              source=source.astype(np.uint8), conf=conf, support=support, nx=nrm[:, 0], ny=nrm[:, 1], nz=nrm[:, 2])
    open_frac = float(np.mean([v["open_water"] for v in per_view.values()])) if per_view else 0.0
    medium = _medium_init(out_dir, meta, pts, C, o_ok, open_rgb, far_rgb, open_frac)

    agg = lambda key: float(np.mean([v[key] for v in per_view.values() if v[key] is not None])) if per_view else None
    cat = lambda a: np.concatenate(a) if a else np.zeros(0)
    summary = {
        "config": asdict(cfg), "grid": asdict(g), "K": Km.tolist(), "views": views, "aux_views": aux, "have_prior": have_prior,
        "points": {"sift": int(len(sp_xyz)), "completed": int(len(cx)), "verified": int(len(vx)),
                   "verified_with_normal": int((np.linalg.norm(vn, axis=1) > 0.5).sum()) if len(vn) else 0,
                   "voxel_verified": v_vox, "voxel_completed": c_vox},
        "mean": {k: agg(k) for k in ("verified", "completed", "open_water", "textured", "verified_on_wreck",
                                     "verified_off_wreck", "keypoint_coverage")},
        "agreement_with_sift": {"mvs": _agreement(cat(agree_mvs)),
                                "prior": _agreement(cat(agree_prior)) if have_prior else None,
                                "same_keypoints": {"mvs": _agreement(cat(agree_both_mvs)),
                                                   "prior": _agreement(cat(agree_both_prior))} if have_prior else None},
        "medium_init": medium, "per_view": per_view, "time_s": round(time.time() - t_start, 1)}
    save_json(os.path.join(out_dir, "mvs", "summary.json"), summary)
    a = summary["agreement_with_sift"]["mvs"]
    mean = summary["mean"]
    log(f"mvs: {len(sp_xyz)} SIFT + {len(cx)} completed + {len(vx)} verified points; pixels verified "
        f"{100 * mean['verified']:.0f}% (wreck {100 * (mean['verified_on_wreck'] or 0):.0f}%), completed "
        f"{100 * mean['completed']:.0f}%; vs SIFT depths: median {100 * a.get('median_rel_err', np.nan):.2f}% "
        f"(p90 {100 * a.get('p90_rel_err', np.nan):.2f}%) on {100 * (mean['keypoint_coverage'] or 0):.0f}% of keypoints")
    return summary
