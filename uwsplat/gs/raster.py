"""Rasterisation backends.

"gsplat": tiled CUDA rasteriser (training / evaluation on the GPU).
"torch" : dense pure-PyTorch reference with the same projection (EWA, eps2d=0.3,
          pixel centres at +0.5) and the same `meta` fields. It is O(N * pixels)
          and only meant for CPU smoke tests on tiny images.
Both render `colors` (C, N, D) plus expected depth as an extra channel ("RGB+ED")
and return (render (C, H, W, D+1), alphas (C, H, W, 1), meta).
"""
import torch
import torch.nn.functional as F


def rasterize(means, quats, scales, opacities, colors, viewmats, Ks, width, height, backend="gsplat",
              absgrad=False, near_plane=0.01):
    if backend == "gsplat":
        from gsplat import rasterization
        return rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height,
                             near_plane=near_plane, render_mode="RGB+ED", packed=False, absgrad=absgrad)
    return torch_rasterize(means, quats, scales, opacities, colors, viewmats, Ks, width, height, near_plane)


def _quat_to_rotmat(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def torch_rasterize(means, quats, scales, opacities, colors, viewmats, Ks, W, H, near_plane=0.01, eps2d=0.3):
    assert viewmats.shape[0] == 1, "reference rasteriser renders one camera at a time"
    Rc, tc, K = viewmats[0, :3, :3], viewmats[0, :3, 3], Ks[0]
    pc = means @ Rc.T + tc
    z = pc[:, 2]
    valid = z > near_plane
    zs = torch.where(valid, z, torch.ones_like(z))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # (1, N, 2): this exact tensor is exposed in meta so that retain_grad() sees its gradient
    means2d = torch.stack([fx * pc[:, 0] / zs + cx, fy * pc[:, 1] / zs + cy], -1)[None]
    Rg = _quat_to_rotmat(quats)
    M = Rg * scales[:, None, :]
    cov = Rc @ (M @ M.transpose(1, 2)) @ Rc.T
    # gsplat's EWA Jacobian, evaluated with x/z, y/z clamped to 1.3x the field of view
    tan_x, tan_y = 0.5 * W / fx, 0.5 * H / fy
    tx = zs * torch.clamp(pc[:, 0] / zs, -(cx / fx + 0.3 * tan_x), (W - cx) / fx + 0.3 * tan_x)
    ty = zs * torch.clamp(pc[:, 1] / zs, -(cy / fy + 0.3 * tan_y), (H - cy) / fy + 0.3 * tan_y)
    J = torch.zeros(len(means), 2, 3, dtype=means.dtype, device=means.device)
    J[:, 0, 0], J[:, 0, 2] = fx / zs, -fx * tx / zs ** 2
    J[:, 1, 1], J[:, 1, 2] = fy / zs, -fy * ty / zs ** 2
    cov2 = J @ cov @ J.transpose(1, 2) + eps2d * torch.eye(2, dtype=means.dtype, device=means.device)
    det = cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] ** 2
    conic = torch.stack([cov2[:, 1, 1], -cov2[:, 0, 1], cov2[:, 0, 0]], -1) / det.clamp_min(1e-12)[:, None]
    radius = torch.ceil(3.33 * torch.sqrt(torch.stack([cov2[:, 0, 0], cov2[:, 1, 1]], -1).clamp_min(0))).detach()
    m2 = means2d[0].detach()
    inside = ((m2 + radius > 0) & (m2 - radius < torch.tensor([W, H], dtype=m2.dtype, device=m2.device))).all(-1)
    valid = valid & (det > 0) & inside

    yy, xx = torch.meshgrid(torch.arange(H, device=means.device) + 0.5, torch.arange(W, device=means.device) + 0.5, indexing="ij")
    pix = torch.stack([xx, yy], -1).reshape(-1, 2)
    order = torch.argsort(torch.where(valid, z, torch.full_like(z, 1e10)))
    d = pix[None] - means2d[0][order][:, None]
    a, b, c = conic[order].unbind(-1)
    sigma = 0.5 * (a[:, None] * d[..., 0] ** 2 + c[:, None] * d[..., 1] ** 2) + b[:, None] * d[..., 0] * d[..., 1]
    alpha = (opacities[order][:, None] * torch.exp(-sigma)).clamp(max=0.999)
    alpha = torch.where((sigma >= 0) & (alpha >= 1.0 / 255.0) & valid[order][:, None], alpha, torch.zeros_like(alpha))
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[:1]), 1.0 - alpha[:-1]], 0), 0)
    w = alpha * T  # (N, P)
    feats = torch.cat([colors[0][order], z[order][:, None]], -1)  # (N, D + 1)
    out = w.T @ feats
    acc = w.sum(0)[:, None]
    out = torch.cat([out[:, :-1], out[:, -1:] / acc.clamp_min(1e-10)], -1)
    radii = torch.where(valid[:, None], radius, torch.zeros_like(radius))
    meta = {"means2d": means2d, "radii": radii[None].int(), "width": W, "height": H, "n_cameras": 1,
            "gaussian_ids": None, "depths": z[None]}
    return out.reshape(1, H, W, -1), acc.reshape(1, H, W, 1), meta
