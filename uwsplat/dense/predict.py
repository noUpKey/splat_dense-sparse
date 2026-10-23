"""Pointmap inference on SfM-selected view pairs.

Submerged3D frames have non-square pixels (fy/fx ~ 0.84), while the network was
trained on square pixels, so every frame is resampled to square pixels before
tokenisation: a 1280x720 frame becomes 512x341 and is centre-cropped to 512x336
(21x32 tokens). `NetGrid` keeps the exact mapping back to full-resolution pixels.
"""
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .weights import load_pointmap_net


@dataclass
class NetGrid:
    W: int      # full-resolution size
    H: int
    w: int      # network input size
    h: int
    h_pre: int  # resized height before the centre crop
    sx: float   # net = full * s - offset
    sy: float
    oy: float

    def to_net(self, uv: np.ndarray) -> np.ndarray:
        return uv * np.array([self.sx, self.sy]) - np.array([0.0, self.oy])

    def K_net(self, K: np.ndarray) -> np.ndarray:
        Kn = K.copy()
        Kn[0] *= self.sx
        Kn[1] *= self.sy
        Kn[1, 2] -= self.oy
        return Kn


def make_grid(W: int, H: int, pixel_aspect: float, width: int = 512, patch: int = 16) -> NetGrid:
    h_pre = int(round(width * H / W / pixel_aspect))
    h = (h_pre // patch) * patch
    return NetGrid(W, H, width, h, h_pre, width / W, h_pre / H, float((h_pre - h) // 2))


def prepare(img: np.ndarray, grid: NetGrid, color_norm: str = "none") -> torch.Tensor:
    if color_norm == "stretch":
        lo, hi = np.percentile(img.reshape(-1, 3), [0.5, 99.5], axis=0)
        img = np.clip((img - lo) / np.maximum(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)
    im = Image.fromarray(img).resize((grid.w, grid.h_pre), Image.LANCZOS)
    x = np.asarray(im)[int(grid.oy):int(grid.oy) + grid.h]
    return torch.from_numpy(x.copy()).permute(2, 0, 1).float() / 127.5 - 1.0


def select_pairs(obs_img: np.ndarray, obs_pt: np.ndarray, views: list, k: int = 4) -> list:
    """For each view, its k most covisible other views (shared SfM points)."""
    views = list(views)
    vis = {v: set(obs_pt[obs_img == v].tolist()) for v in views}
    pairs = []
    for i in views:
        score = sorted(((len(vis[i] & vis[j]), j) for j in views if j != i), reverse=True)
        pairs += [(i, j) for s, j in score[:k] if s > 0]
    return pairs


class DensePredictor:
    def __init__(self, weights: str, device: str = "cuda", dtype: str = "bf16"):
        self.device = device
        self.net = load_pointmap_net(weights, device)
        self.amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype) if device.startswith("cuda") else None

    @torch.no_grad()
    def __call__(self, images: list, pairs: list, grid: NetGrid, color_norm: str = "none", batch: int = 4):
        """Returns (pts (P,h,w,3), conf (P,h,w)) for the first view of every pair, on CPU."""
        views = sorted({i for p in pairs for i in p})
        ctx = torch.autocast("cuda", dtype=self.amp) if self.amp else torch.autocast("cpu", enabled=False)
        feats, pos = {}, None
        with ctx:
            for v in views:  # each image is encoded exactly once
                f, pos = self.net.encode(prepare(images[v], grid, color_norm)[None].to(self.device))
                feats[v] = f[0]
            pts, conf = [], []
            for s in range(0, len(pairs), batch):
                chunk = pairs[s:s + batch]
                fr = torch.stack([feats[i] for i, _ in chunk])
                fs = torch.stack([feats[j] for _, j in chunk])
                p = pos[:1].expand(len(chunk), -1, -1)
                a, c = self.net.pointmap(fr, fs, p, p, grid.h, grid.w)
                pts.append(a.float().cpu())
                conf.append(c.float().cpu())
        return torch.cat(pts), torch.cat(conf)
