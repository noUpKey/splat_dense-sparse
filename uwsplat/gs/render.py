"""Single-pass underwater rendering.

Per Gaussian i and camera c:  J_i = SH(d_ic) + 0.5 (restored radiance),
  I_i = J_i exp(-beta_D r_ic) + B_inf(d_ic) (1 - exp(-beta_B r_ic))   (underwater colour).
[I_i, J_i] are rasterised together with expected depth (7 channels, one pass),
then the open-water background (1 - alpha) B_inf(d_pixel) is composited.
"""
import torch

from .model import sh_coeffs
from .raster import rasterize
from .sh import eval_sh


def render(params, view, medium=None, sh_degree: int = 0, backend: str = "gsplat", absgrad: bool = False,
           appearance=None, bg_color=None) -> dict:
    means = params["means"]
    v = means - view.campos
    r = v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    d = v / r
    J = torch.clamp_min(eval_sh(sh_degree, sh_coeffs(params), d) + 0.5, 0.0)
    I = medium.compose(J, r, d) if medium is not None else J
    H, W = view.hw
    scales, opacities = torch.exp(params["scales"]), torch.sigmoid(params["opacities"])
    scatter = getattr(medium, "scatter", None) if medium is not None else None
    if scatter is not None:  # forward scattering: range-dependent blur, energy-preserving (gs/scatter.py)
        scales, opacities = scatter.apply(means, params["quats"], scales, opacities, view.viewmat, view.K)
    out, alpha, meta = rasterize(means, params["quats"], scales, opacities,
                                 torch.cat([I, J], -1)[None], view.viewmat[None], view.K[None], W, H,
                                 backend=backend, absgrad=absgrad)
    out, alpha = out[0], alpha[0]
    bg = medium.binf(view.dirs) if medium is not None else bg_color.expand(H, W, 3)
    image = out[..., :3] + (1.0 - alpha) * bg
    if appearance is not None and view.train_id >= 0:
        image = appearance(image, view.train_id)
    return {"image": image, "restored": out[..., 3:6], "depth": out[..., 6:7], "alpha": alpha, "background": bg,
            "meta": meta}
