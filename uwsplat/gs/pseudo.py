"""Pseudo-view geometric regulariser: the verified MVS cloud checks the Gaussians from poses nobody trained on.

Why. With 16-30 training views the photometric loss is satisfied long before the geometry is right: training
PSNR reaches ~35 dB while held-out views stay ~13 dB lower. Gaussians that only have to look right from the
training cameras can sit in front of the surface (floaters that cover a training view's view-specific content)
or drift along the training rays once the depth priors have decayed; from a new pose they land in the wrong
place. The training-view depth priors cannot see this: along a training ray a floater and the surface can
render the same expected depth.

What. Every `every` steps a camera is interpolated between two consecutive training cameras (centre lerp,
rotation slerp, t ~ U(-extrap, 1 + extrap)), rendered at reduced resolution (depth and alpha only matter),
and the rendered depth is compared with the depth of the MVS-verified points seen from that camera:

    L_pv = mean over checked pixels of |log D_render - log D_points|

Selective by construction:
  * only MVS-verified points are targets (photometrically checked in >= 2 real views, 0.2-0.5 % depth error
    against SIFT); network-only points and SIFT points are not used, open water has no points;
  * a pixel is checked only if a point projects there, it is the front-most point (z-buffer), it is not
    behind another surface (5x5 min-filter test: points seen through gaps of a nearer surface are dropped),
    and the render covers it (alpha > 0.5);
  * no image is rendered or compared at the pseudo pose: there is no photo there, so only geometry is
    constrained and the appearance stays free.
The weight does not decay (the targets do not get less reliable during training), unlike the training-view
depth priors.
"""
import math
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


def _rot_to_quat(R):
    """(3, 3) -> unit quaternion (w, x, y, z)."""
    t = R.trace()
    if t > 0:
        s = math.sqrt(float(t) + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(torch.argmax(torch.diagonal(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(float(1.0 + R[i, i] - R[j, j] - R[k, k])) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = torch.tensor([float(x) for x in q])
    return q / q.norm()


def _quat_to_rot(q):
    w, x, y, z = q.tolist()
    return torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(q0, q1, t):
    d = float((q0 * q1).sum())
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / q.norm()
    th = math.acos(min(d, 1.0))
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def interpolate_pose(vm0, vm1, t):
    """World->camera matrices (4, 4) of two cameras -> the camera at fraction t (centre lerp, rotation slerp;
    t outside [0, 1] extrapolates along the same path)."""
    R0, R1 = vm0[:3, :3].cpu().double(), vm1[:3, :3].cpu().double()
    C0, C1 = -R0.T @ vm0[:3, 3].cpu().double(), -R1.T @ vm1[:3, 3].cpu().double()
    R = _quat_to_rot(slerp(_rot_to_quat(R0), _rot_to_quat(R1), t)).double()
    C = (1 - t) * C0 + t * C1
    vm = torch.eye(4, dtype=torch.float64)
    vm[:3, :3], vm[:3, 3] = R, -R @ C
    return vm.float().to(vm0.device), C.float().to(vm0.device)


def points_depth(X, viewmat, K, W, H, near=1e-3, win=5, tol=0.02):
    """Z-buffer depth of points X (N, 3) in a camera, with points behind a nearer surface removed:
    a pixel keeps its depth only if it is within `tol` (relative) of the minimum depth in its win x win
    neighbourhood. Returns (H, W) depth, 0 where there is no target."""
    pc = X @ viewmat[:3, :3].T + viewmat[:3, 3]
    z = pc[:, 2]
    ok = z > near
    u = (pc[:, 0] / z.clamp_min(near)) * K[0, 0] + K[0, 2]
    v = (pc[:, 1] / z.clamp_min(near)) * K[1, 1] + K[1, 2]
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = v[ok].long() * W + u[ok].long()
    zb = torch.full((H * W,), float("inf"), device=X.device)
    zb = zb.scatter_reduce(0, idx, z[ok], reduce="amin").view(H, W)
    finite = torch.isfinite(zb)
    big = torch.where(finite, zb, torch.full_like(zb, 1e9))
    zmin = -F.max_pool2d(-big[None, None], win, stride=1, padding=win // 2)[0, 0]
    keep = finite & (zb <= zmin * (1 + tol))
    return torch.where(keep, zb, torch.zeros_like(zb))


class PseudoViews:
    def __init__(self, train_views, verified_xyz, scale=0.25, extrap=0.25, seed=0, max_points=1_500_000):
        views = sorted(train_views, key=lambda v: v.name)
        self.pairs = [(views[k], views[k + 1]) for k in range(len(views) - 1)]
        dev = views[0].viewmat.device
        X = np.asarray(verified_xyz, dtype=np.float32)
        if len(X) > max_points:
            X = X[np.random.default_rng(seed).choice(len(X), max_points, replace=False)]
        self.X = torch.from_numpy(X).to(dev)
        H0, W0 = views[0].hw
        self.W, self.H = max(8, int(round(W0 * scale))), max(8, int(round(H0 * scale)))
        K = views[0].K.clone()
        K[0] *= self.W / W0
        K[1] *= self.H / H0
        self.K = K
        v, u = torch.meshgrid(torch.arange(self.H, device=dev) + 0.5, torch.arange(self.W, device=dev) + 0.5,
                              indexing="ij")
        rays = torch.stack([u, v, torch.ones_like(u)], -1) @ torch.linalg.inv(K).T
        self.ray_norm = rays.norm(dim=-1, keepdim=True)
        self.rays_unit = rays / self.ray_norm
        self.extrap = extrap
        self.rng = random.Random(seed)
        self._dummy = torch.zeros(self.H, self.W, 1, device=dev)

    def __len__(self):
        return len(self.pairs)

    def sample(self):
        """A pseudo camera (duck-typed View for render()) and its target depth map (H, W), 0 = unchecked."""
        a, b = self.pairs[self.rng.randrange(len(self.pairs))]
        t = self.rng.uniform(-self.extrap, 1 + self.extrap)
        vm, C = interpolate_pose(a.viewmat, b.viewmat, t)
        view = SimpleNamespace(name=f"pseudo:{a.stem}-{b.stem}@{t:.2f}", viewmat=vm, K=self.K, campos=C,
                               image=self._dummy, dirs=self.rays_unit @ vm[:3, :3], ray_norm=self.ray_norm,
                               train_id=-1, hw=(self.H, self.W))
        with torch.no_grad():
            target = points_depth(self.X, vm, self.K, self.W, self.H)
        return view, target


def pseudo_depth_loss(depth, alpha, target):
    """Mean |log rendered depth - log target| over pixels with a target that the render covers."""
    m = ((target > 0)[..., None] & (alpha.detach() > 0.5)).float()
    err = (torch.log(depth.clamp_min(1e-6)) - torch.log(target[..., None].clamp_min(1e-6))).abs()
    return (m * err).sum() / m.sum().clamp_min(1.0), m.mean()
