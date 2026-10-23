"""Camera models.

All pixel coordinates in uwsplat follow the COLMAP / gsplat convention: the centre
of the top-left pixel is (0.5, 0.5). OpenCV remap/undistort functions index pixel
centres at integers, so `opencv_K` shifts the principal point by -0.5.

Submerged3D frames are 3:2 photos resized to 16:9, i.e. pixels are not square
(fy/fx ~ 0.84). The distorted model therefore keeps separate fx and fy, with the
radial terms acting in (square) normalised camera coordinates.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class DistortedCamera:
    """COLMAP OPENCV model with zero tangential terms: fx, fy, cx, cy, k1, k2."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def opencv_K(self) -> np.ndarray:
        K = self.K.copy()
        K[0, 2] -= 0.5
        K[1, 2] -= 0.5
        return K

    @property
    def dist(self) -> np.ndarray:
        return np.array([self.k1, self.k2, 0.0, 0.0])

    @property
    def params(self) -> np.ndarray:
        return np.array([self.fx, self.fy, self.cx, self.cy, self.k1, self.k2])

    def with_params(self, p) -> "DistortedCamera":
        return DistortedCamera(self.width, self.height, *map(float, p))

    def distort(self, xn: np.ndarray) -> np.ndarray:
        r2 = np.sum(xn * xn, -1, keepdims=True)
        return xn * (1.0 + self.k1 * r2 + self.k2 * r2 * r2)

    def undistort(self, xd: np.ndarray, iters: int = 30) -> np.ndarray:
        x = xd.copy()
        for _ in range(iters):
            r2 = np.sum(x * x, -1, keepdims=True)
            x = xd / (1.0 + self.k1 * r2 + self.k2 * r2 * r2)
        return x

    def pixel_to_normalized(self, uv: np.ndarray) -> np.ndarray:
        xd = (uv - np.array([self.cx, self.cy])) / np.array([self.fx, self.fy])
        return self.undistort(xd)

    def normalized_to_pixel(self, xn: np.ndarray) -> np.ndarray:
        return self.distort(xn) * np.array([self.fx, self.fy]) + np.array([self.cx, self.cy])

    def to_dict(self) -> dict:
        d = asdict(self)
        return {"model": "OPENCV", **{k: (v if isinstance(v, int) else float(v)) for k, v in d.items()}}

    @staticmethod
    def from_dict(d: dict) -> "DistortedCamera":
        return DistortedCamera(int(d["width"]), int(d["height"]), *(float(d[k]) for k in ("fx", "fy", "cx", "cy", "k1", "k2")))


@dataclass
class PinholeCamera:
    """Undistorted pinhole camera used by the dense and splatting stages."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def pixel_aspect(self) -> float:
        """fy / fx; 1.0 for square pixels."""
        return self.fy / self.fx

    def scaled(self, width: int, height: int) -> "PinholeCamera":
        sx, sy = width / self.width, height / self.height
        return PinholeCamera(width, height, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy)

    def to_dict(self) -> dict:
        return {"model": "PINHOLE", **asdict(self)}

    @staticmethod
    def from_dict(d: dict) -> "PinholeCamera":
        return PinholeCamera(int(d["width"]), int(d["height"]), float(d["fx"]), float(d["fy"]),
                             float(d["cx"]), float(d["cy"]))
