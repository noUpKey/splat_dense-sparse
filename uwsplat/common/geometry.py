"""Rotation / projection helpers shared by all stages (numpy, batched)."""
import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """(..., 3) -> (..., 3, 3) cross-product matrices."""
    z = np.zeros_like(v[..., 0])
    return np.stack([
        np.stack([z, -v[..., 2], v[..., 1]], -1),
        np.stack([v[..., 2], z, -v[..., 0]], -1),
        np.stack([-v[..., 1], v[..., 0], z], -1),
    ], -2)


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues formula, (..., 3) rotation vectors -> (..., 3, 3) matrices."""
    w = np.asarray(w, dtype=np.float64)
    theta = np.linalg.norm(w, axis=-1)[..., None, None]
    k = skew(w / np.maximum(theta[..., 0], 1e-12))
    R = np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)
    small = theta[..., 0, 0] < 1e-8
    if np.any(small):
        R[small] = np.eye(3) + skew(w[small])
    return R


def so3_log(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices -> (..., 3) rotation vectors."""
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(theta)[..., None]
    w = np.where(s > 1e-8, v * (theta[..., None] / np.maximum(2.0 * s, 1e-12)), 0.5 * v)
    return w


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Single 3x3 rotation -> quaternion (w, x, y, z), COLMAP convention."""
    m = R
    tr = np.trace(m)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q * np.sign(q[0]) if q[0] != 0 else q


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World-to-camera (R, t) -> camera centre(s) in world coordinates."""
    return -np.einsum("...ji,...j->...i", R, t)


def ray_angle(c1: np.ndarray, c2: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Triangulation angle (radians) at points X seen from centres c1, c2."""
    a = c1 - X
    b = c2 - X
    cos = np.sum(a * b, -1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def triangulate_dlt(P: np.ndarray, x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Batched multi-view DLT.

    P: (M, V, 3, 4) projection matrices in normalized camera coordinates,
    x: (M, V, 2) normalized (undistorted) image points, valid: (M, V) bool.
    Returns (M, 3) points; rows with fewer than two valid views are NaN.
    """
    A = np.concatenate([
        x[..., 0:1] * P[..., 2, :] - P[..., 0, :],
        x[..., 1:2] * P[..., 2, :] - P[..., 1, :],
    ], axis=1)  # (M, 2V, 4)
    w = np.concatenate([valid, valid], axis=1).astype(np.float64)[..., None]
    A = A * w
    A = A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)
    _, _, vt = np.linalg.svd(A)
    Xh = vt[:, -1]
    X = Xh[:, :3] / np.where(np.abs(Xh[:, 3:4]) < 1e-12, 1e-12, Xh[:, 3:4])
    X[valid.sum(1) < 2] = np.nan
    return X
