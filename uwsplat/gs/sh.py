"""Real spherical-harmonics basis (degrees 0-3), same convention as 3DGS / gsplat."""
import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def sh_basis(degree: int, dirs: torch.Tensor) -> torch.Tensor:
    """dirs (..., 3) unit vectors -> (..., (degree + 1) ** 2) basis values."""
    x, y, z = dirs.unbind(-1)
    out = [torch.full_like(x, C0)]
    if degree > 0:
        out += [-C1 * y, C1 * z, -C1 * x]
    if degree > 1:
        xx, yy, zz = x * x, y * y, z * z
        out += [C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * zz - xx - yy), C2[3] * x * z, C2[4] * (xx - yy)]
        if degree > 2:
            out += [C3[0] * y * (3 * xx - yy), C3[1] * x * y * z, C3[2] * y * (4 * zz - xx - yy),
                    C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
                    C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)]
    return torch.stack(out, -1)


def eval_sh(degree: int, coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """coeffs (N, K, 3) with K >= (degree+1)^2, dirs (N, 3) -> (N, 3)."""
    b = sh_basis(degree, dirs)
    return torch.einsum("nk,nkc->nc", b, coeffs[:, :b.shape[-1]])


def rgb_to_sh(rgb):
    return (rgb - 0.5) / C0
