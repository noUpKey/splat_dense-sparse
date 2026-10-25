"""SfM post-processing: similarity normalisation, undistortion and export.

Outputs (under the scene output directory):
  images/<stem>.png           undistorted frames (lossless, used as training + test GT)
  sfm/scene.json              pinhole camera, per-image pose / split, normalisation, stats
  sfm/points.npz              3D points + observations in undistorted pixel coordinates
  sparse/0/*.txt              COLMAP text model (PINHOLE) so 3DGS-style baselines run on it
"""
import os

import cv2
import numpy as np

from ..common.camera import DistortedCamera, PinholeCamera
from ..common.geometry import camera_center, rotmat_to_quat
from ..common.io import log, save_json, save_rgb
from .bundle import reprojection_errors
from .reconstruct import SfMResult


def test_split(names: list, every: int = 8) -> list:
    """Standard LLFF/3DGS hold-out: every `every`-th image (sorted by name) is a test view."""
    order = {n: k for k, n in enumerate(sorted(names))}
    return ["test" if order[n] % every == 0 else "train" for n in names]


def normalize(res: SfMResult):
    """Centre on the mean camera centre and scale so that the median camera-point range is 1."""
    C = camera_center(res.R, res.t)
    center = C[res.registered].mean(0)
    ranges = np.linalg.norm(res.points[res.obs_pt] - C[res.obs_img], axis=1)
    s = 1.0 / float(np.median(ranges))
    t = s * (res.t + np.einsum("nij,j->ni", res.R, center))
    return s * (res.points - center), t, {"center": center.tolist(), "scale": s}


def undistorted_camera(cam: DistortedCamera, n: int = 200) -> PinholeCamera:
    """Pinhole camera with the same size and centre whose every pixel is covered by the source."""
    W, H = cam.width, cam.height
    u, v = np.linspace(0.5, W - 0.5, n), np.linspace(0.5, H - 0.5, n)
    edges = {
        "l": np.stack([np.full(n, 0.5), v], 1), "r": np.stack([np.full(n, W - 0.5), v], 1),
        "t": np.stack([u, np.full(n, 0.5)], 1), "b": np.stack([u, np.full(n, H - 0.5)], 1),
    }
    xn = {k: cam.pixel_to_normalized(p) for k, p in edges.items()}
    half_x = min(-xn["l"][:, 0].max(), xn["r"][:, 0].min())
    half_y = min(-xn["t"][:, 1].max(), xn["b"][:, 1].min())
    s = max((W / 2.0) / (cam.fx * half_x), (H / 2.0) / (cam.fy * half_y))
    return PinholeCamera(W, H, cam.fx * s, cam.fy * s, W / 2.0, H / 2.0)


def undistort_maps(cam: DistortedCamera, pin: PinholeCamera):
    u, v = np.meshgrid(np.arange(pin.width) + 0.5, np.arange(pin.height) + 0.5)
    xn = np.stack([(u - pin.cx) / pin.fx, (v - pin.cy) / pin.fy], -1)
    src = cam.normalized_to_pixel(xn) - 0.5  # OpenCV indexes pixel centres at integers
    return src[..., 0].astype(np.float32), src[..., 1].astype(np.float32)


def max_distortion_px(cam: DistortedCamera, pin: PinholeCamera) -> float:
    mx, my = undistort_maps(cam, pin)
    u, v = np.meshgrid(np.arange(pin.width), np.arange(pin.height))
    return float(np.max(np.hypot(mx - u, my - v)))


def export_sfm(res: SfMResult, images: list, out_dir: str, scene: str, test_every: int = 8,
               min_distortion_px: float = 0.3) -> dict:
    cam = res.camera
    if cam.k1 == 0.0 and cam.k2 == 0.0:
        # pinhole model (sfm.distortion=False, images already undistorted): keep the images exactly as they are.
        # undistorted_camera() would still zoom by half a pixel (it covers pixel centres), i.e. resample them.
        pin, disp = PinholeCamera(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy), 0.0
    else:
        pin = undistorted_camera(cam)
        disp = max_distortion_px(cam, pin)
        if disp < min_distortion_px:
            pin = PinholeCamera(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy)
    log(f"undistortion: max displacement {disp:.1f}px -> pinhole fx={pin.fx:.1f} fy={pin.fy:.1f}")
    mx, my = undistort_maps(cam, pin)
    files = []
    for name, img in zip(res.names, images):
        out = img if disp < min_distortion_px else cv2.remap(img, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        fname = os.path.join("images", os.path.splitext(name)[0] + ".png")
        save_rgb(os.path.join(out_dir, fname), out)
        files.append(fname)

    X, t, norm = normalize(res)
    xn = cam.pixel_to_normalized(res.obs_uv)
    uv = xn * np.array([pin.fx, pin.fy]) + np.array([pin.cx, pin.cy])
    split = test_split(res.names, test_every)
    meta = {
        "scene": scene, "width": pin.width, "height": pin.height, "camera": pin.to_dict(),
        "distorted_camera": cam.to_dict(), "max_distortion_px": disp, "normalization": norm,
        "test_every": test_every, "sfm_stats": res.stats,
        "images": [{"name": n, "file": f, "registered": bool(r), "split": s,
                    "R": Ri.tolist(), "t": ti.tolist()}
                   for n, f, r, s, Ri, ti in zip(res.names, files, res.registered, split, res.R, t)],
    }
    save_json(os.path.join(out_dir, "sfm", "scene.json"), meta)
    np.savez_compressed(os.path.join(out_dir, "sfm", "points.npz"), xyz=X.astype(np.float32), rgb=res.colors,
                        obs_img=res.obs_img.astype(np.int32), obs_pt=res.obs_pt.astype(np.int32),
                        obs_uv=uv.astype(np.float32), obs_rgb=res.obs_rgb)
    write_colmap_text(os.path.join(out_dir, "sparse", "0"), pin, res, X, t, files, uv)
    return meta


def write_colmap_text(d: str, pin: PinholeCamera, res: SfMResult, X, t, files, uv):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cameras.txt"), "w") as f:
        f.write(f"1 PINHOLE {pin.width} {pin.height} {pin.fx} {pin.fy} {pin.cx} {pin.cy}\n")
    obs_idx = [np.where(res.obs_img == i)[0] for i in range(len(res.names))]
    track = [[] for _ in range(len(X))]
    with open(os.path.join(d, "images.txt"), "w") as f:
        for i, (fname, Ri, ti) in enumerate(zip(files, res.R, t)):
            if not res.registered[i]:
                continue
            q = rotmat_to_quat(Ri)
            f.write(f"{i + 1} {q[0]} {q[1]} {q[2]} {q[3]} {ti[0]} {ti[1]} {ti[2]} 1 {os.path.basename(fname)}\n")
            pts = []
            for k, o in enumerate(obs_idx[i]):
                pts.append(f"{uv[o, 0]:.3f} {uv[o, 1]:.3f} {res.obs_pt[o] + 1}")
                track[res.obs_pt[o]].append(f"{i + 1} {k}")
            f.write(" ".join(pts) + "\n")
    err = reprojection_errors(res.camera.params, res.R, res.t, res.points, res.obs_img, res.obs_pt, res.obs_uv)
    err = np.bincount(res.obs_pt, err, len(X)) / np.maximum(np.bincount(res.obs_pt, minlength=len(X)), 1)
    with open(os.path.join(d, "points3D.txt"), "w") as f:
        for j, (p, c) in enumerate(zip(X, res.colors)):
            f.write(f"{j + 1} {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} {err[j]:.4f} {' '.join(track[j])}\n")
