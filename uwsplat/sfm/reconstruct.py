"""Incremental SIFT structure-from-motion: the sparse, high-precision branch.

COLMAP-like pipeline, re-implemented:
  CLAHE+RootSIFT -> mutual ratio matching -> MAGSAC fundamental verification -> tracks
  -> self-calibration of fx and fy from the fundamental matrices (no EXIF available)
  -> initial pair with many well-triangulated points at large parallax
  -> repeat { PnP-RANSAC registration, DLT triangulation, global BA, outlier filtering }
  -> retry failed images, final retriangulation and full-intrinsics BA.
"""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..common.camera import DistortedCamera
from ..common.geometry import camera_center, triangulate_dlt
from ..common.io import log
from .bundle import bundle_adjust, project
from .features import extract_features
from .matching import match_all
from .tracks import Tracks


@dataclass
class SfMConfig:
    max_features: int = 8192
    clahe_clip: float = 2.0
    contrast_threshold: float = 0.04
    match_ratio: float = 0.8
    match_max_angle: float = 0.7
    f_thresh_px: float = 1.5
    min_pair_inliers: int = 15
    init_min_points: int = 80
    init_good_angle_deg: float = 8.0
    pnp_thresh_px: float = 4.0
    pnp_min_inliers: int = 24
    pnp_min_ratio: float = 0.15
    tri_max_err_px: float = 4.0
    tri_min_angle_deg: float = 1.5
    final_max_err_px: float = 3.0
    loss_scale_px: float = 2.0
    refine_intrinsics_from: int = 4
    distortion: bool = True           # False: pinhole (k1 = k2 = 0), for images that are already undistorted, so
                                      # they reach training and evaluation unchanged (e.g. SeaThru-NeRF prepared
                                      # with scripts/prepare_seathru.py)
    ba_iters: int = 15
    final_ba_iters: int = 60
    device: str = "cpu"


@dataclass
class SfMResult:
    names: list
    camera: DistortedCamera
    R: np.ndarray            # (n_img, 3, 3) world-to-camera
    t: np.ndarray            # (n_img, 3)
    registered: np.ndarray   # (n_img,) bool
    points: np.ndarray       # (n_pts, 3)
    colors: np.ndarray       # (n_pts, 3) uint8
    obs_img: np.ndarray      # (n_obs,) image index of each observation
    obs_pt: np.ndarray       # (n_obs,) point index
    obs_uv: np.ndarray       # (n_obs, 2) measured keypoint (distorted pixels)
    obs_rgb: np.ndarray      # (n_obs, 3) uint8 colour under the keypoint
    stats: dict = field(default_factory=dict)


def estimate_intrinsics(pairs, width, height, n_best=60):
    """Self-calibrate (fx, fy) with a centred principal point.

    For the true K, E = K^T F K has two equal non-zero singular values; the cost
    sum_pairs (s1 - s2) / (s1 + s2) is minimised over a (fx, fy/fx) grid. Pairs
    explained by a homography (pure rotation / planar) are down-weighted.
    """
    s = max(width, height)
    fxs = np.geomspace(0.25 * s, 3.0 * s, 160)
    aspects = np.linspace(0.6, 1.4, 81)
    FX, AS = np.meshgrid(fxs, aspects, indexing="ij")
    K = np.zeros(FX.shape + (3, 3))
    K[..., 0, 0], K[..., 1, 1] = FX, FX * AS
    K[..., 0, 2], K[..., 1, 2], K[..., 2, 2] = width / 2.0, height / 2.0, 1.0
    weight = lambda p: len(p.idx_i) * max(1.0 - p.h_ratio, 0.05)
    best = sorted(pairs, key=weight, reverse=True)[:n_best]
    cost = np.zeros(FX.shape)
    for p in best:
        E = np.einsum("...ji,jk,...kl->...il", K, p.F, K)
        sv = np.linalg.svd(E, compute_uv=False)
        cost += weight(p) * (sv[..., 0] - sv[..., 1]) / np.maximum(sv[..., 0] + sv[..., 1], 1e-12)
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    return float(fxs[i]), float(fxs[i] * aspects[j])


class IncrementalSfM:
    def __init__(self, images: list, names: list, cfg: SfMConfig = SfMConfig()):
        self.cfg, self.names = cfg, names
        self.n_img = len(images)
        H, W = images[0].shape[:2]
        t0 = time.time()
        self.feats = [extract_features(im, cfg.max_features, cfg.clahe_clip, contrast_threshold=cfg.contrast_threshold)
                      for im in images]
        log(f"SIFT: {np.mean([len(f) for f in self.feats]):.0f} keypoints / image")
        self.pairs = match_all(self.feats, cfg.match_ratio, cfg.match_max_angle, cfg.f_thresh_px,
                               cfg.min_pair_inliers, cfg.device)
        log(f"matching: {len(self.pairs)} verified pairs of {self.n_img * (self.n_img - 1) // 2}")
        self.tracks = T = Tracks([len(f) for f in self.feats], self.pairs)
        log(f"tracks: {T.n} (mean length {T.length.mean():.2f}, {T.n_conflicting} inconsistent dropped)")
        offs = np.concatenate([[0], np.cumsum([len(f) for f in self.feats])])
        self.obs_uv = np.concatenate([f.xy for f in self.feats])[offs[T.obs_img] + T.obs_kp].astype(np.float64)
        self.obs_rgb = np.concatenate([f.rgb for f in self.feats])[offs[T.obs_img] + T.obs_kp]
        self.img_obs = [np.where(T.obs_img == i)[0] for i in range(self.n_img)]
        fx, fy = estimate_intrinsics(self.pairs, W, H)
        log(f"self-calibration: fx={fx:.1f} fy={fy:.1f} (fy/fx={fy / fx:.3f})")
        self.f_init = (fx, fy)
        self.cam = DistortedCamera(W, H, fx, fy, W / 2.0, H / 2.0)
        self.intrinsics = ("fx", "fy", "k1", "k2") if cfg.distortion else ("fx", "fy")
        self.X = np.full((T.n, 3), np.nan)
        self.obs_ok = np.zeros(len(T.obs_img), bool)
        self.reg = np.zeros(self.n_img, bool)
        self.R = np.tile(np.eye(3), (self.n_img, 1, 1))
        self.t = np.zeros((self.n_img, 3))
        self.ref = None
        self.t_front = time.time() - t0

    # ----------------------------------------------------------------- helpers
    @property
    def f_mean(self):
        return 0.5 * (self.cam.fx + self.cam.fy)

    def _padded(self, tids):
        T = self.tracks
        start, L = T.ptr[tids], T.length[tids]
        V = int(L.max()) if len(L) else 1
        idx = start[:, None] + np.arange(V)[None]
        valid = np.arange(V)[None] < L[:, None]
        return np.where(valid, idx, 0), valid

    def _errors(self, obs):
        """Pixel reprojection error of observations (inf for missing points / behind camera)."""
        T = self.tracks
        tr = T.obs_track[obs]
        err = np.full(len(obs), np.inf)
        ok = np.isfinite(self.X[tr, 0]) & self.reg[T.obs_img[obs]]
        if ok.any():
            o = obs[ok]
            uv, P, *_ = project(self.cam.params, self.R, self.t, self.X, T.obs_img[o], T.obs_track[o])
            e = np.linalg.norm(uv - self.obs_uv[o], axis=1)
            e[P[:, 2] <= 1e-6] = np.inf
            err[ok] = e
        return err

    def _max_angles(self, tids, obs_mask):
        idx, valid = self._padded(tids)
        valid &= obs_mask[idx]
        C = camera_center(self.R, self.t)[self.tracks.obs_img[idx]]
        rays = C - self.X[tids][:, None]
        rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-12)
        cos = np.einsum("mai,mbi->mab", rays, rays)
        pair_ok = valid[:, :, None] & valid[:, None, :]
        cos = np.where(pair_ok, cos, 1.0)
        return np.degrees(np.arccos(np.clip(cos.min(axis=(1, 2)), -1.0, 1.0)))

    # ------------------------------------------------------------ main stages
    def _initial_pair(self):
        cfg = self.cfg
        best, best_score = None, 0.0
        for p in sorted(self.pairs, key=lambda p: -len(p.idx_i))[:40]:
            if len(p.idx_i) < cfg.init_min_points:
                continue
            x1 = self.cam.pixel_to_normalized(self.feats[p.i].xy[p.idx_i].astype(np.float64))
            x2 = self.cam.pixel_to_normalized(self.feats[p.j].xy[p.idx_j].astype(np.float64))
            thr = cfg.f_thresh_px / self.f_mean
            E, mask = cv2.findEssentialMat(x1, x2, np.eye(3), method=cv2.RANSAC, prob=0.9999, threshold=thr)
            if E is None or mask is None:
                continue
            _, Rr, tr, mpose = cv2.recoverPose(E[:3], x1, x2, np.eye(3), mask=mask.copy())
            P = np.stack([np.hstack([np.eye(3), np.zeros((3, 1))]), np.hstack([Rr, tr])])
            Xp = triangulate_dlt(np.broadcast_to(P, (len(x1), 2, 3, 4)), np.stack([x1, x2], 1),
                                 np.ones((len(x1), 2), bool))
            z1, X2 = Xp[:, 2], Xp @ Rr.T + tr.ravel()
            e1 = np.linalg.norm(self.cam.normalized_to_pixel(Xp[:, :2] / Xp[:, 2:3]) - self.cam.normalized_to_pixel(x1), axis=1)
            e2 = np.linalg.norm(self.cam.normalized_to_pixel(X2[:, :2] / X2[:, 2:3]) - self.cam.normalized_to_pixel(x2), axis=1)
            c2 = -Rr.T @ tr.ravel()
            ang = np.degrees(np.arccos(np.clip(np.sum(-Xp * (c2 - Xp), 1) / np.maximum(
                np.linalg.norm(Xp, axis=1) * np.linalg.norm(c2 - Xp, axis=1), 1e-12), -1, 1)))
            good = (z1 > 0) & (X2[:, 2] > 0) & (e1 < cfg.tri_max_err_px) & (e2 < cfg.tri_max_err_px) & (ang > cfg.tri_min_angle_deg)
            if good.sum() < cfg.init_min_points:
                continue
            med = float(np.median(ang[good]))
            score = good.sum() * min(1.0, med / cfg.init_good_angle_deg)
            if score > best_score:
                best, best_score = (p.i, p.j, Rr, tr.ravel(), int(good.sum()), med), score
        if best is None:
            raise RuntimeError("no valid initial pair found")
        i, j, Rr, tr, n, med = best
        log(f"initial pair {self.names[i]} / {self.names[j]}: {n} points, median angle {med:.1f} deg")
        self.ref = i
        self.reg[[i, j]] = True
        self.R[j], self.t[j] = Rr, tr

    def _register(self, i, min_inliers, thresh_px):
        T = self.tracks
        obs = self.img_obs[i]
        obs = obs[np.isfinite(self.X[T.obs_track[obs], 0])]
        if len(obs) < min_inliers:
            return False
        X3 = self.X[T.obs_track[obs]]
        xn = self.cam.pixel_to_normalized(self.obs_uv[obs])
        ok, rvec, tvec, inl = cv2.solvePnPRansac(X3, xn, np.eye(3), None, iterationsCount=10000,
                                                 reprojectionError=thresh_px / self.f_mean,
                                                 confidence=0.9999, flags=cv2.SOLVEPNP_AP3P)
        if not ok or inl is None or len(inl) < max(min_inliers, self.cfg.pnp_min_ratio * len(obs)):
            return False
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(X3[inl], xn[inl], np.eye(3), None, rvec, tvec)
        self.R[i], self.t[i] = cv2.Rodrigues(rvec)[0], tvec.ravel()
        self.reg[i] = True
        log(f"registered {self.names[i]}: {len(inl)}/{len(obs)} PnP inliers ({self.reg.sum()} images)")
        return True

    def _triangulate_missing(self):
        cfg, T = self.cfg, self.tracks
        reg_obs = self.reg[T.obs_img]
        n_reg_obs = np.bincount(T.obs_track, reg_obs, T.n)
        tids = np.where(~np.isfinite(self.X[:, 0]) & (n_reg_obs >= 2))[0]
        if len(tids) == 0:
            return 0
        idx, valid = self._padded(tids)
        for _ in range(2):  # triangulate, drop outlying views, re-triangulate
            valid &= reg_obs[idx]
            img = T.obs_img[idx]
            P = np.concatenate([self.R[img], self.t[img][..., None]], -1)
            xn = self.cam.pixel_to_normalized(self.obs_uv[idx])
            X = triangulate_dlt(P, xn, valid)
            Xc = np.einsum("mvij,mj->mvi", self.R[img], np.nan_to_num(X)) + self.t[img]
            z = Xc[..., 2]
            uv = self.cam.normalized_to_pixel(Xc[..., :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[..., None])
            err = np.linalg.norm(uv - self.obs_uv[idx], axis=-1)
            valid &= (z > 1e-6) & (err < cfg.tri_max_err_px) & np.isfinite(X[:, :1])
        keep = valid.sum(1) >= 2
        self.X[tids[keep]] = X[keep]
        return int(keep.sum())

    def _refresh(self, max_err):
        """Re-evaluate every observation and drop unstable points."""
        cfg, T = self.cfg, self.tracks
        all_obs = np.arange(len(T.obs_img))
        self.obs_ok = self._errors(all_obs) < max_err
        n_ok = np.bincount(T.obs_track, self.obs_ok, T.n)
        self.X[n_ok < 2] = np.nan
        tids = np.where(np.isfinite(self.X[:, 0]))[0]
        if len(tids):
            ang = self._max_angles(tids, self.obs_ok)
            self.X[tids[ang < cfg.tri_min_angle_deg]] = np.nan
        self.obs_ok &= np.isfinite(self.X[T.obs_track, 0])

    def _bundle(self, iters, refine):
        T = self.tracks
        obs = np.where(self.obs_ok)[0]
        uniq, pt = np.unique(T.obs_track[obs], return_inverse=True)
        free = self.reg.copy()
        free[self.ref] = False
        cp, self.R, self.t, Xn, info = bundle_adjust(
            self.cam.params, self.R, self.t, self.X[uniq], T.obs_img[obs], pt, self.obs_uv[obs], free,
            refine=refine, fixed_ref=self.ref, max_iters=iters, loss_scale=self.cfg.loss_scale_px)
        self.cam = self.cam.with_params(cp)
        self.X[uniq] = Xn
        return info

    def _step(self, iters=None):
        cfg = self.cfg
        self._triangulate_missing()
        self._refresh(cfg.tri_max_err_px)
        refine = self.intrinsics if self.reg.sum() >= cfg.refine_intrinsics_from else ()
        info = self._bundle(iters or cfg.ba_iters, refine)
        self._refresh(cfg.tri_max_err_px)
        return info

    def run(self) -> SfMResult:
        cfg, T = self.cfg, self.tracks
        t0 = time.time()
        self._initial_pair()
        self._step()
        failed = set()
        for relaxed in (False, True):
            min_inl = cfg.pnp_min_inliers if not relaxed else max(12, cfg.pnp_min_inliers // 2)
            thr = cfg.pnp_thresh_px * (1.5 if relaxed else 1.0)
            while True:
                n_corr = []
                for i in np.where(~self.reg)[0]:
                    o = self.img_obs[i]
                    n_corr.append((int(np.isfinite(self.X[T.obs_track[o], 0]).sum()), i))
                n_corr.sort(reverse=True)
                progress = False
                for _, i in n_corr:
                    if (i, relaxed) in failed:
                        continue
                    if self._register(i, min_inl, thr):
                        progress = True
                        self._step()
                        failed.clear()  # new structure: every image may now register
                        break
                    failed.add((i, relaxed))
                if not progress:
                    break
        # final polishing
        self._triangulate_missing()
        self._refresh(cfg.tri_max_err_px)
        self._bundle(cfg.final_ba_iters, self.intrinsics)
        self._refresh(cfg.final_max_err_px)
        self._triangulate_missing()
        self._refresh(cfg.final_max_err_px)
        info = self._bundle(cfg.final_ba_iters, self.intrinsics)
        self._refresh(cfg.final_max_err_px)
        return self._result(info, time.time() - t0)

    def _result(self, info, t_map):
        T = self.tracks
        keep = np.isfinite(self.X[:, 0])
        new_id = np.cumsum(keep) - 1
        obs = np.where(self.obs_ok)[0]
        obs_pt = new_id[T.obs_track[obs]]
        err = self._errors(obs)
        rgb = np.zeros((keep.sum(), 3))
        np.add.at(rgb, obs_pt, self.obs_rgb[obs].astype(np.float64))
        n_obs = np.bincount(obs_pt, minlength=keep.sum())
        rgb = (rgb / np.maximum(n_obs, 1)[:, None]).round().astype(np.uint8)
        stats = {
            "registered": int(self.reg.sum()), "images": self.n_img,
            "unregistered": [self.names[i] for i in np.where(~self.reg)[0]],
            "points": int(keep.sum()), "observations": int(len(obs)),
            "mean_track_length": float(n_obs.mean()) if len(n_obs) else 0.0,
            "mean_reproj_error_px": float(err.mean()) if len(err) else 0.0,
            "verified_pairs": len(self.pairs), "self_calibration_fx_fy": list(self.f_init),
            "camera": self.cam.to_dict(), "time_features_matching_s": round(self.t_front, 1),
            "time_mapping_s": round(t_map, 1),
        }
        log(f"SfM done: {stats['registered']}/{self.n_img} images, {stats['points']} points, "
            f"track {stats['mean_track_length']:.2f}, reproj {stats['mean_reproj_error_px']:.3f}px, "
            f"fx={self.cam.fx:.1f} fy={self.cam.fy:.1f} k1={self.cam.k1:.4f} k2={self.cam.k2:.4f}")
        return SfMResult(self.names, self.cam, self.R.copy(), self.t.copy(), self.reg.copy(), self.X[keep].copy(),
                         rgb, T.obs_img[obs], obs_pt, self.obs_uv[obs], self.obs_rgb[obs], stats)


def run_sfm(images: list, names: list, cfg: SfMConfig = SfMConfig()) -> SfMResult:
    return IncrementalSfM(images, names, cfg).run()
