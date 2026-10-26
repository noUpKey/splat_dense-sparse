"""Underwater image formation (revised Akkaynak-Treibitz model), applied per Gaussian.

  I = J * exp(-beta_D * r) + B_inf(d) * (1 - exp(-beta_B * r))

r is the range along the viewing ray and d the world-space ray direction.
Compared with SeaSplat's per-pixel version (attenuation/backscatter networks fed
with a min-max normalised rendered depth map), only 9 + 9 scalars remain:
  * beta_D, beta_B: one coefficient per colour channel, in scene units (the
    scene is normalised to median range 1, so they are comparable across views);
    DeepSeeColor's double exponential, residual backscatter term and white-balance
    gain are dropped: they are not identifiable once Gaussian colours are free;
  * B_inf: the veiling light. Real open water is brighter towards the surface
    and darker downwards, so B_inf is a sigmoid of a degree-1 SH in the ray
    direction (12 numbers) instead of one constant colour.
Applying the model to each Gaussian (colour c_i at range r_i) and compositing
over a B_inf background equals the volumetric integral of the medium between
and behind semi-transparent Gaussians, so no separate depth pass is needed.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sh import C0, sh_basis


def _inv_softplus(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    return x + torch.log(-torch.expm1(-x))


class WaterMedium(nn.Module):
    def __init__(self, binf, beta_d, beta_b, binf_degree: int = 1):
        super().__init__()
        self.binf_degree = binf_degree
        self.beta_d_raw = nn.Parameter(_inv_softplus(beta_d))
        self.beta_b_raw = nn.Parameter(_inv_softplus(beta_b))
        coeffs = torch.zeros((binf_degree + 1) ** 2, 3)
        b = torch.as_tensor(np.clip(binf, 0.01, 0.99), dtype=torch.float32)
        coeffs[0] = torch.logit(b) / C0
        self.binf_sh = nn.Parameter(coeffs)

    @property
    def beta_d(self):
        return F.softplus(self.beta_d_raw)

    @property
    def beta_b(self):
        return F.softplus(self.beta_b_raw)

    def binf(self, dirs: torch.Tensor) -> torch.Tensor:
        """Veiling light for unit world-space directions (..., 3) -> (..., 3)."""
        return torch.sigmoid(sh_basis(self.binf_degree, dirs) @ self.binf_sh)

    def compose(self, J: torch.Tensor, r: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        """Underwater colour of radiance J seen at range r (..., 1) along dirs."""
        return J * torch.exp(-self.beta_d * r) + self.binf(dirs) * (1.0 - torch.exp(-self.beta_b * r))

    def restore(self, I: torch.Tensor, r: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        """Inverse model, used to initialise Gaussian colours from observed colours."""
        bs = self.binf(dirs) * (1.0 - torch.exp(-self.beta_b * r))
        return (I - bs) * torch.exp(self.beta_d * r)

    def summary(self) -> dict:
        extra = self.scatter.summary() if hasattr(self, "scatter") else {}
        return {**extra, "beta_d": self.beta_d.tolist(), "beta_b": self.beta_b.tolist(),
                "binf_sh": self.binf_sh.detach().cpu().tolist(), "binf_dc": torch.sigmoid(C0 * self.binf_sh[0]).tolist()}
