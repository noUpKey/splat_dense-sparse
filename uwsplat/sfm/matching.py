"""Exhaustive descriptor matching and two-view geometric verification."""
from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
import torch

from .features import Features


@dataclass
class TwoView:
    i: int
    j: int
    idx_i: np.ndarray   # inlier keypoint indices in image i
    idx_j: np.ndarray   # matching keypoint indices in image j
    F: np.ndarray       # fundamental matrix (pixel coordinates, x_j^T F x_i = 0)
    n_putative: int
    h_ratio: float      # fraction of F-inliers explained by a homography (planarity / pure rotation)


def mutual_ratio_match(d1: torch.Tensor, d2: torch.Tensor, ratio: float = 0.8,
                       max_angle: float = 0.7):
    """Mutual nearest neighbours with Lowe's ratio test on descriptor angles (as in COLMAP)."""
    if len(d1) < 2 or len(d2) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    sim = d1 @ d2.T
    top = sim.topk(2, dim=1)
    ang = torch.arccos(top.values.clamp(-1.0, 1.0))
    nn12 = top.indices[:, 0]
    nn21 = sim.argmax(0)
    ok = (ang[:, 0] < max_angle) & (ang[:, 0] < ratio * ang[:, 1])
    ok &= nn21[nn12] == torch.arange(len(d1), device=d1.device)
    i1 = torch.nonzero(ok).squeeze(1)
    return i1.cpu().numpy(), nn12[i1].cpu().numpy()


def verify_pair(f1: Features, f2: Features, i1, i2, f_thresh: float = 1.5,
                min_inliers: int = 15):
    if len(i1) < min_inliers:
        return None
    p1 = f1.xy[i1].astype(np.float64)
    p2 = f2.xy[i2].astype(np.float64)
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.USAC_MAGSAC, f_thresh, 0.9999, 10000)
    if F is None or F.shape != (3, 3) or mask is None:
        return None
    mask = mask.ravel().astype(bool)
    n = int(mask.sum())
    if n < min_inliers:
        return None
    H, hmask = cv2.findHomography(p1[mask], p2[mask], cv2.USAC_MAGSAC, 3.0, maxIters=5000, confidence=0.999)
    h_ratio = float(hmask.sum()) / n if H is not None and hmask is not None else 0.0
    return i1[mask], i2[mask], F, h_ratio


def match_all(feats: list, ratio: float = 0.8, max_angle: float = 0.7, f_thresh: float = 1.5,
              min_inliers: int = 15, device: str = "cpu", pairs=None) -> list:
    descs = [torch.from_numpy(f.desc.astype(np.float32)).to(device) for f in feats]
    out = []
    for i, j in pairs if pairs is not None else combinations(range(len(feats)), 2):
        a, b = mutual_ratio_match(descs[i], descs[j], ratio, max_angle)
        res = verify_pair(feats[i], feats[j], a, b, f_thresh, min_inliers)
        if res is not None:
            out.append(TwoView(i, j, res[0], res[1], res[2], len(a), res[3]))
    return out
