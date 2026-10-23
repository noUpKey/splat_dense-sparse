"""Per-pixel modality routing.

wreck       : pixels with dense support of triangulated SIFT tracks -> sparse SfM points
water_surf  : remaining pixels whose dense depth is confident and multi-view consistent
              (sand, seabed, haze-dominated far structure)  -> dense points
open_water  : everything else (water column, no reliable surface) -> rendered by the medium

Multi-view consistency alone does not reject the water column: neighbouring frames
have small baselines and hallucinate nearly the same 'wall' in open water. The
network's confidence does: on Isro, conf >= 1.2 keeps 94% of SIFT-keypoint pixels
but only 3% of smooth featureless open-water pixels (water sits at the floor, 1.0).
"""
import cv2
import numpy as np


def sift_support(uv: np.ndarray, W: int, H: int, cell_px: float = 24.0) -> np.ndarray:
    """Keypoint density >= one track per cell_px x cell_px area (Gaussian sigma = cell/2).
    A single isolated keypoint cannot pass, so stray matches on particles do not create 'wreck'."""
    cnt = np.zeros((H, W), np.float32)
    ij = np.clip(np.floor(uv).astype(int), 0, [W - 1, H - 1])
    np.add.at(cnt, (ij[:, 1], ij[:, 0]), 1.0)
    dens = cv2.GaussianBlur(cnt, (0, 0), sigmaX=cell_px / 2.0, borderType=cv2.BORDER_CONSTANT)
    mask = (dens >= 1.0 / cell_px ** 2).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(cell_px) | 1, int(cell_px) | 1))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k).astype(bool)


def local_std(gray: np.ndarray, k: int = 7) -> np.ndarray:
    g = gray.astype(np.float32)
    mu = cv2.blur(g, (k, k))
    return np.sqrt(np.clip(cv2.blur(g * g, (k, k)) - mu * mu, 0, None))


def route(wreck_full: np.ndarray, depth: np.ndarray, n_cons: np.ndarray, conf: np.ndarray, gray_net: np.ndarray,
          grid, min_consistent: int = 2, min_conf: float = 1.2, rescue_conf: float = 1.05, rescue_std: float = 2.0):
    """Masks at network resolution: (wreck, water_surface, open_water).

    A consistent dense pixel is a surface if the network is confident, or if it is moderately
    confident and the image is textured there (open water is nearly noise-flat, std < ~1.2/255).
    """
    wreck = cv2.resize(wreck_full.astype(np.uint8), (grid.w, grid.h_pre), interpolation=cv2.INTER_NEAREST)
    wreck = wreck[int(grid.oy):int(grid.oy) + grid.h].astype(bool)
    textured = local_std(gray_net) >= rescue_std
    surf = (depth > 0) & (n_cons >= min_consistent) & ((conf >= min_conf) | ((conf >= rescue_conf) & textured))
    return wreck, ~wreck & surf, ~wreck & ~surf
