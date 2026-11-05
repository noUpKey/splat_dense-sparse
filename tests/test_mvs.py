"""MVS verification stage and the new training pieces, on synthetic data with known answers.

python -m tests.test_mvs
  1. plane sweep: a slanted textured plane seen by 5 cameras through water (range-dependent attenuation
     and backscatter, different exposure per frame, sensor noise). The NCC sweep must recover the depth
     of the textured part to < 0.5 % and refuse the flat (water-like) part.
  2. depth_normals: normals of the true depth map match the plane normal.
  3. normal_to_quat: the local z axis of each initial Gaussian is rotated onto its normal.
  4. RUSplatting ESL: zero on flat images, stopped at depth edges, weights detached by default.
  5. _bilinear_rgb: > 32767 lookups (cv2.remap limit) give the same colours as small batches.
"""
import numpy as np
import torch

from uwsplat.fusion.fuse import _bilinear_rgb
from uwsplat.gs.losses import esl_smoothness
from uwsplat.gs.model import normal_to_quat
from uwsplat.gs.raster import _quat_to_rotmat
from uwsplat.mvs.run import depth_normals, relative_pose
from uwsplat.mvs.sweep import best_depth, ncc_volume, window_std


def render_plane(K, C, h, w, tex, extent, z0=2.0, slope=0.35, beta=(0.6, 0.25), binf=0.35, gain=1.0, flat_x=None):
    """Grey image of the plane z = z0 + slope * x seen from centre C (R = I), through water.
    Texture tex (n, n) covers [-extent, extent]^2 of the plane's (x, y); x > flat_x is textureless."""
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    d = np.stack([u + 0.5, v + 0.5, np.ones_like(u)], -1) @ np.linalg.inv(K).T
    s = (z0 + slope * C[0] - C[2]) / (d[..., 2] - slope * d[..., 0])
    P = C + s[..., None] * d
    n = tex.shape[0]
    ij = (P[..., :2] + extent) / (2 * extent) * (n - 1)
    i0 = np.clip(np.floor(ij).astype(int), 0, n - 2)
    f = np.clip(ij - i0, 0, 1)
    T = (tex[i0[..., 1], i0[..., 0]] * (1 - f[..., 0]) * (1 - f[..., 1]) + tex[i0[..., 1], i0[..., 0] + 1] * f[..., 0] * (1 - f[..., 1])
         + tex[i0[..., 1] + 1, i0[..., 0]] * (1 - f[..., 0]) * f[..., 1] + tex[i0[..., 1] + 1, i0[..., 0] + 1] * f[..., 0] * f[..., 1])
    if flat_x is not None:
        T = np.where(P[..., 0] > flat_x, 0.5, T)
    r = np.linalg.norm(P - C, axis=-1)
    I = gain * (T * np.exp(-beta[0] * r) + binf * (1 - np.exp(-beta[1] * r)))
    return I * 255.0, P[..., 2] - C[2]


def test_sweep():
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    tex = gaussian_filter(rng.random((900, 900)), 1.2)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    h, w = 120, 200
    K = np.array([[160.0, 0, w / 2], [0, 135.0, h / 2], [0, 0, 1]])  # non-square pixels, like Submerged3D
    xs = [0.0, -0.12, 0.12, -0.25, 0.25]
    gains = [1.0, 0.85, 1.15, 0.9, 1.1]  # per-frame exposure
    imgs, depths = [], []
    for x, gn in zip(xs, gains):
        C = np.array([x, 0.0, 0.0])
        I, Z = render_plane(K, C, h, w, tex, 4.0, gain=gn, flat_x=0.9)
        imgs.append(I + rng.normal(0, 0.6, I.shape))
        depths.append(Z)
    R = np.stack([np.eye(3)] * 5)
    t = np.stack([-np.array([x, 0.0, 0.0]) for x in xs])
    ref = torch.tensor(imgs[0], dtype=torch.float32)
    srcs = torch.tensor(np.stack(imgs[1:]), dtype=torch.float32)
    rel = torch.tensor(np.stack([relative_pose(R, t, 0, j) for j in range(1, 5)]), dtype=torch.float32)
    lo, hi = 1.0, 4.5
    d0 = torch.full((h, w), float(np.sqrt(lo * hi)))
    half = torch.full((h, w), float(0.5 * np.log(hi / lo)))
    ncc, tau = ncc_volume(ref, srcs, torch.tensor(K, dtype=torch.float32), rel, d0, half, 160, window=7)
    d, s, valid = best_depth(ncc, tau, d0, half, topk=2, min_ncc=0.5)
    textured = window_std(ref, 7) >= 2.0
    Z = torch.tensor(depths[0], dtype=torch.float32)
    tex_region = torch.tensor(np.isfinite(depths[0])) & textured
    ok = valid & tex_region
    err = (d[ok] / Z[ok] - 1).abs()
    cover = ok.float().sum() / tex_region.float().sum()
    print(f"sweep: textured coverage {100 * cover:.1f}%, median depth error {100 * err.median():.3f}%, "
          f"p90 {100 * err.quantile(0.9):.3f}%")
    assert cover > 0.8 and err.median() < 0.005 and err.quantile(0.9) < 0.02
    # the flat part (x > 0.9 on the plane) is noise-only: the texture gate must reject it
    P_x = (np.arange(w) + 0.5 - K[0, 2]) / K[0, 0] * depths[0]
    flat = torch.tensor(P_x > 1.0)
    frac = (textured & flat).float().sum() / flat.float().sum()
    print(f"sweep: flat region passing the texture gate {100 * frac:.2f}%")
    assert frac < 0.02
    return d, Z, K


def test_normals(Z, K, slope=0.35):
    n = depth_normals(Z.numpy().astype(np.float64), K, step=2, max_jump=0.03)
    truth = np.array([slope, 0.0, -1.0]) / np.linalg.norm([slope, 0.0, -1.0])
    m = np.linalg.norm(n, axis=-1) > 0.5
    ang = np.degrees(np.arccos(np.clip(n[m] @ truth, -1, 1)))
    print(f"normals: {100 * m.mean():.1f}% estimated, median angle error {np.median(ang):.3f} deg")
    assert m.mean() > 0.9 and np.median(ang) < 0.5


def test_quats():
    rng = np.random.default_rng(1)
    n = rng.normal(size=(1000, 3))
    n[:3] = [[0, 0, 1], [0, 0, -1], [1, 0, 0]]
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    Rm = _quat_to_rotmat(torch.tensor(normal_to_quat(n), dtype=torch.float32)).numpy()
    err = np.abs(Rm[:, :, 2] - n).max()
    print(f"normal_to_quat: max |R e_z - n| = {err:.2e}")
    assert err < 1e-5


def test_esl():
    H, W = 32, 40
    alpha = torch.ones(H, W, 1)
    depth = torch.ones(H, W, 1)
    flat = torch.full((H, W, 3), 0.4)
    assert float(esl_smoothness(flat, depth, alpha)) == 0.0
    img = flat.clone()
    img[:, W // 2:] = 0.8  # vertical image edge
    smooth = float(esl_smoothness(img, depth, alpha))
    d_edge = depth.clone()
    d_edge[:, W // 2:] = 2.0  # same place: a depth edge (relative jump ~0.67 of the mean)
    stopped = float(esl_smoothness(img, d_edge, alpha))
    print(f"ESL: image edge on a smooth surface {smooth:.5f}, at a depth edge {stopped:.5f}")
    assert stopped < 0.1 * smooth
    d = d_edge.clone().requires_grad_(True)
    im = img.clone().requires_grad_(True)
    esl_smoothness(im, d, alpha, detach=True).backward()
    assert d.grad is None or float(d.grad.abs().sum()) == 0.0
    assert float(im.grad.abs().sum()) > 0
    d2 = d_edge.clone().requires_grad_(True)
    esl_smoothness(img, d2, alpha, detach=False).backward()
    assert float(d2.grad.abs().sum()) > 0  # RUSplatting's form: depth receives a gradient


def test_bilinear_rgb():
    rng = np.random.default_rng(2)
    img = (rng.random((90, 160, 3)) * 255).astype(np.uint8)
    uv = rng.random((50_000, 2)) * np.array([160, 90])
    big = _bilinear_rgb(img, uv)
    small = np.concatenate([_bilinear_rgb(img, uv[k:k + 1000]) for k in range(0, len(uv), 1000)])
    assert big.shape == (50_000, 3) and np.array_equal(big, small)
    assert _bilinear_rgb(img, np.zeros((0, 2))).shape == (0, 3)


if __name__ == "__main__":
    torch.manual_seed(0)
    d, Z, K = test_sweep()
    test_normals(Z, K)
    test_quats()
    test_esl()
    test_bilinear_rgb()
    print("all MVS / loss tests passed")
