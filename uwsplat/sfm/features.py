"""SIFT front-end of the sparse (shipwreck) branch.

Underwater images are low-contrast and hazy, so SIFT runs on a CLAHE-equalised
grey image; descriptors are RootSIFT (L1-normalise + sqrt), the same
normalisation COLMAP uses, which makes the dot product a Hellinger kernel.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Features:
    xy: np.ndarray     # (N, 2) float32 keypoints, (0.5, 0.5) = centre of top-left pixel
    desc: np.ndarray   # (N, 128) float16 RootSIFT, unit L2 norm
    scale: np.ndarray  # (N,) float32 keypoint size in pixels
    rgb: np.ndarray    # (N, 3) uint8 colour under the keypoint

    def __len__(self):
        return len(self.xy)


def extract_features(img: np.ndarray, max_features: int = 8192, clahe_clip: float = 2.0,
                     clahe_grid: int = 8, contrast_threshold: float = 0.04,
                     edge_threshold: float = 10.0) -> Features:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if clahe_clip > 0:
        gray = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid)).apply(gray)
    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=contrast_threshold,
                           edgeThreshold=edge_threshold)
    kps, desc = sift.detectAndCompute(gray, None)
    if desc is None or len(kps) == 0:
        return Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float16),
                        np.zeros(0, np.float32), np.zeros((0, 3), np.uint8))
    xy = np.array([k.pt for k in kps], np.float32)
    scale = np.array([k.size for k in kps], np.float32)
    desc = desc.astype(np.float32)
    desc /= desc.sum(1, keepdims=True) + 1e-7
    desc = np.sqrt(desc)
    h, w = gray.shape
    rc = np.clip(np.round(xy).astype(int), 0, [w - 1, h - 1])
    rgb = img[rc[:, 1], rc[:, 0]]
    return Features(xy + 0.5, desc.astype(np.float16), scale, rgb)
