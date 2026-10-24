"""Scene loading for training / evaluation (cameras, images, multimodal priors)."""
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..common.camera import PinholeCamera
from ..common.geometry import camera_center
from ..common.io import load_json, load_rgb, read_ply
from ..dense.predict import NetGrid


@dataclass
class View:
    idx: int
    name: str
    stem: str
    split: str
    viewmat: torch.Tensor          # (4, 4) world -> camera
    K: torch.Tensor                # (3, 3)
    campos: torch.Tensor           # (3,)
    image: torch.Tensor            # (H, W, 3) in [0, 1]
    dirs: torch.Tensor             # (H, W, 3) unit world-space ray directions
    ray_norm: torch.Tensor         # (H, W, 1) range = z-depth * ray_norm
    train_id: int = -1
    prior_depth: Optional[torch.Tensor] = None  # (H, W) dense prior (water region), 0 = none
    prior_w: Optional[torch.Tensor] = None
    kp_uv: Optional[torch.Tensor] = None        # (n, 2) SIFT keypoints of the wreck
    kp_z: Optional[torch.Tensor] = None
    wreck: Optional[torch.Tensor] = None        # (H, W) bool, for region-wise metrics

    @property
    def hw(self):
        return self.image.shape[0], self.image.shape[1]


def _dense_to_image(depth, conf, grid: NetGrid, W, H, s):
    """Nearest resampling of a network-grid map onto the (downscaled) image grid."""
    u = (np.arange(W) + 0.5) / s * grid.sx
    v = (np.arange(H) + 0.5) / s * grid.sy - grid.oy
    g = torch.stack(torch.meshgrid(torch.tensor(v / grid.h * 2 - 1), torch.tensor(u / grid.w * 2 - 1),
                                   indexing="ij")[::-1], -1).float()[None]
    m = torch.from_numpy(np.stack([depth, conf]))[None].float()
    out = F.grid_sample(m, g, mode="nearest", padding_mode="zeros", align_corners=False)[0]
    return out[0], out[1]


GEOMETRIES = ("fusion", "mvs")  # v1: SIFT wreck + network water cloud | v2: + MVS-verified dense cloud


def _prior_grid(out_dir: str, geometry: str):
    """Grid of the per-view prior maps: the network grid (fusion) or the MVS grid (mvs)."""
    path = os.path.join(out_dir, "mvs" if geometry == "mvs" else "dense", "summary.json")
    return NetGrid(**load_json(path)["grid"]) if os.path.exists(path) else None


class SceneData:
    def __init__(self, out_dir: str, device="cuda", downscale: float = 1.0, load_priors: bool = True,
                 geometry: str = "fusion"):
        if geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
        self.out_dir, self.device, self.geometry = out_dir, device, geometry
        meta = load_json(os.path.join(out_dir, "sfm", "scene.json"))
        pin = PinholeCamera.from_dict(meta["camera"])
        W, H = int(round(pin.width / downscale)), int(round(pin.height / downscale))
        s = W / pin.width
        K = pin.scaled(W, H).K
        grid = _prior_grid(out_dir, geometry) if load_priors else None
        v, u = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing="ij")
        rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
        ray_norm = np.linalg.norm(rays, axis=-1, keepdims=True)
        self.views, self.train, self.test = [], [], []
        n_train = 0
        for i, im in enumerate(meta["images"]):
            if not im["registered"]:
                continue
            R, t = np.array(im["R"]), np.array(im["t"])
            vm = np.eye(4)
            vm[:3, :3], vm[:3, 3] = R, t
            img = load_rgb(os.path.join(out_dir, im["file"]))
            if downscale != 1.0:
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            dirs = (rays / ray_norm) @ R  # camera -> world (R^T applied to row vectors)
            stem = os.path.splitext(im["name"])[0]
            view = View(i, im["name"], stem, im["split"], torch.tensor(vm, dtype=torch.float32, device=device),
                        torch.tensor(K, dtype=torch.float32, device=device),
                        torch.tensor(camera_center(R, t), dtype=torch.float32, device=device),
                        torch.from_numpy(img).float().div(255).to(device),
                        torch.tensor(dirs, dtype=torch.float32, device=device),
                        torch.tensor(ray_norm, dtype=torch.float32, device=device))
            for gdir in dict.fromkeys((geometry,) + GEOMETRIES):  # same SIFT-support masks in both stages
                mpath = os.path.join(out_dir, gdir, "masks", stem + ".png")
                if os.path.exists(mpath):
                    m = cv2.resize(cv2.imread(mpath, cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_NEAREST)
                    view.wreck = torch.from_numpy(m > 127).to(device)
                    break
            ppath = os.path.join(out_dir, geometry, "priors", stem + ".npz")
            if im["split"] == "train":
                view.train_id = n_train
                n_train += 1
                if grid is not None and os.path.exists(ppath):
                    pr = np.load(ppath)
                    d, c = _dense_to_image(pr["dense_depth"], pr["dense_conf"], grid, W, H, s)
                    view.prior_depth, view.prior_w = d.to(device), c.to(device)
                    view.kp_uv = torch.from_numpy(pr["kp_uv"] * s).float().to(device)
                    view.kp_z = torch.from_numpy(pr["kp_z"]).float().to(device)
                self.train.append(view)
            else:
                self.test.append(view)
            self.views.append(view)
        C = torch.stack([v.campos for v in self.train])
        # 3DGS convention: extent = 1.1 x radius of the training camera centres
        self.scene_scale = float(1.1 * (C - C.mean(0)).norm(dim=-1).max())
        self.width, self.height = W, H
        self.meta = meta

    def points(self, kind: str = "fused"):
        """Initial point cloud of this geometry: xyz, rgb, source tag and normals (zeros where unknown).

        kind: 'fused' (everything), 'sparse' (SIFT only), 'dense' (everything but SIFT),
        'verified' (SIFT + MVS-verified, i.e. without the network-only points; mvs geometry)."""
        ply = read_ply(os.path.join(self.out_dir, self.geometry, "fused.ply"))
        xyz = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float64)
        rgb = np.stack([ply["red"], ply["green"], ply["blue"]], 1) / 255.0
        src = ply["source"].astype(np.uint8)
        nrm = np.stack([ply["nx"], ply["ny"], ply["nz"]], 1).astype(np.float64) if "nx" in ply else np.zeros_like(xyz)
        keep = {"fused": np.ones(len(src), bool), "sparse": src == 0, "dense": src >= 1,
                "verified": src != 1}[kind]
        return xyz[keep], rgb[keep], src[keep], nrm[keep]
