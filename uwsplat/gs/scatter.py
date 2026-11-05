"""Forward scattering: blur that grows with range, applied per Gaussian and per view.

Light scattered at small angles on its way from a surface to the camera spreads the surface's image into a blur
whose angular width grows with the path length r (Jaffe-McGlamery: kernel ~ exp(-B r f), width ~ r). Without it, a
Gaussian seen sharply from near views and blurred from far views has to compromise, which shows up as renders that
are softer than the photos. Here each Gaussian keeps its sharp shape and the renderer widens it for each view:

  angular blur sigma_a = kappa * r                (radians; kappa learned per scene, one scalar)
  3D isotropic widening b = sigma_a * z = kappa * r * z     -> Sigma_3D' = Sigma_3D + b^2 I
  energy conservation: opacity' = opacity * sqrt(det Sigma_2D / det Sigma_2D')

The 2D covariances are computed exactly as the rasteriser projects them (EWA, + 0.3 px^2), so the integrated
footprint (opacity x area) of every Gaussian is unchanged: forward scatter moves light around, it does not remove it.
Because b^2 I is added in 3D and the rotation is unchanged, Sigma_3D' = R diag(s^2 + b^2) R^T: the rasteriser just
gets scales sqrt(s^2 + b^2). No CUDA change.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .raster import _quat_to_rotmat

EPS2D = 0.3  # the rasteriser's screen-space dilation (px^2)


class ForwardScatter(nn.Module):
    def __init__(self, kappa_init: float = 3e-4):
        super().__init__()
        self.kappa_raw = nn.Parameter(torch.tensor(math.log(math.expm1(kappa_init))))
        self.active = True

    @property
    def kappa(self):
        return F.softplus(self.kappa_raw)

    def apply(self, means, quats, scales, opacities, viewmat, K):
        """Widened scales and energy-preserving opacities for one camera (means / scales world units)."""
        if not self.active:
            return scales, opacities
        Rc, tc = viewmat[:3, :3], viewmat[:3, 3]
        pc = means @ Rc.T + tc
        z = pc[:, 2].clamp_min(1e-2)
        r = pc.norm(dim=-1)
        b2 = (self.kappa * r * z) ** 2                                   # (N,)
        fx, fy = K[0, 0], K[1, 1]
        J = torch.zeros(len(means), 2, 3, dtype=means.dtype, device=means.device)
        J[:, 0, 0], J[:, 0, 2] = fx / z, -fx * pc[:, 0] / z ** 2
        J[:, 1, 1], J[:, 1, 2] = fy / z, -fy * pc[:, 1] / z ** 2
        JW = J @ Rc                                                       # (N, 2, 3)
        Rg = _quat_to_rotmat(quats)
        M = Rg * scales[:, None, :]
        cov2 = JW @ (M @ M.transpose(1, 2)) @ JW.transpose(1, 2)
        cov2 = cov2 + EPS2D * torch.eye(2, dtype=means.dtype, device=means.device)
        JJt = JW @ JW.transpose(1, 2)                                     # projection of b^2 I (W orthonormal)
        cov2n = cov2 + b2[:, None, None] * JJt
        det = lambda c: (c[:, 0, 0] * c[:, 1, 1] - c[:, 0, 1] * c[:, 1, 0]).clamp_min(1e-12)
        ratio = torch.sqrt(det(cov2) / det(cov2n)).clamp(max=1.0)
        return torch.sqrt(scales ** 2 + b2[:, None]), opacities * ratio

    def summary(self) -> dict:
        return {"kappa": float(self.kappa.detach())}
