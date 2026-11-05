"""Geometric-uncertainty splatting (C2): render every Gaussian with its own position uncertainty. No training.

For Gaussian i, the training views that actually see it (accumulated blending weight W_ik > tau, measured by the
rasteriser) give the Fisher information of its position

    F_i = sum_k w_ik J_ik^T J_ik / sigma_px^2 + Lambda0_i,        w_ik = min(1, W_ik / W0),

with J_ik the 2x3 projection Jacobian at its centre (pixels per world unit) and Lambda0_i a weak range prior along
the mean viewing ray (standard deviation sigma_z x mean range). Its position covariance is P_i = F_i^-1 (3x3, world
frame): small across the rays of the views that see it, long along the ray of a Gaussian that one view sees.

Render-time variants for any camera c (J = that camera's Jacobian, eps = the rasteriser's 0.3 px^2 dilation):

    base     the trained model, unchanged
    c2       Sigma' = Sigma + kappa P_i;  alpha' = alpha sqrt(det(J Sigma J^T + eps) / det(J Sigma' J^T + eps))
    fade     Sigma unchanged;             alpha' = the c2 factor (the opacity loss of c2 without its blur)
    c2_det3  Sigma' as c2;                alpha' = alpha sqrt(det Sigma / det Sigma')   (3D factor, as Mip / Robust GS)
    mip      Sigma' = Sigma + sigma_px^2 (min_k z_ik / f_k)^2 I over the views that see it, 3D factor: Mip-Splatting's
             3D filter applied after training (the best-view isotropic special case of P_i)
    cull     the trained model without the Gaussians that no training view sees (control: c2 and fade drop those
             too, because their prior-only P makes the opacity factor tiny; c2 must beat this to be more than culling)

Properties (proven; checked by tests/test_uncertainty.py): the 2D factor conserves each splat's integrated
contribution in every view; in every training view with w_ik = 1, J P J^T <= sigma_px^2 I, so training views change
by at most sigma_px^2 px^2 per splat. Splats whose new peak opacity falls below 1/255 are dropped from the render
(the rasteriser would skip every one of their pixels anyway).
"""
import os

import numpy as np
import torch

from ..common.io import log, save_json, save_rgb
from ..gs.data import SceneData
from ..gs.model import sh_coeffs
from ..gs.raster import _quat_to_rotmat, rasterize
from ..gs.sh import eval_sh
from .evaluate import load_model
from .heldout import apply_affine, centred_appearance, seen_mask
from .metrics import lpips, psnr, quantize, ssim

EPS2D = 0.3
VARIANTS = ("base", "c2", "fade", "c2_det3", "mip", "cull")


# ----------------------------------------------------------------------------------------------- geometry helpers
def gaussian_cov(quats, scales):
    """(N, 3, 3) world covariances R diag(s^2) R^T."""
    R = _quat_to_rotmat(quats)
    M = R * scales[:, None, :]
    return M @ M.transpose(1, 2)


def proj_jacobian(means, viewmat, K):
    """(J (N, 2, 3) world -> pixels per world unit at each centre, camera-frame points (N, 3))."""
    Rc, tc = viewmat[:3, :3], viewmat[:3, 3]
    pc = means @ Rc.T + tc
    z = pc[:, 2].clamp_min(1e-4)
    J = torch.zeros(len(means), 2, 3, dtype=means.dtype, device=means.device)
    J[:, 0, 0], J[:, 0, 2] = K[0, 0] / z, -K[0, 0] * pc[:, 0] / z ** 2
    J[:, 1, 1], J[:, 1, 2] = K[1, 1] / z, -K[1, 1] * pc[:, 1] / z ** 2
    return J @ Rc.to(means.dtype), pc


def _det2(c):
    return (c[:, 0, 0] * c[:, 1, 1] - c[:, 0, 1] * c[:, 1, 0]).clamp_min(1e-20)


def _det3(c):
    """Closed-form 3x3 determinant (batched linalg on millions of tiny matrices can need huge GPU workspaces)."""
    return (c[:, 0, 0] * (c[:, 1, 1] * c[:, 2, 2] - c[:, 1, 2] * c[:, 2, 1])
            - c[:, 0, 1] * (c[:, 1, 0] * c[:, 2, 2] - c[:, 1, 2] * c[:, 2, 0])
            + c[:, 0, 2] * (c[:, 1, 0] * c[:, 2, 1] - c[:, 1, 1] * c[:, 2, 0]))


def factor_2d(J, S, S2, eps=EPS2D):
    """sqrt(det(J S J^T + eps I) / det(J S2 J^T + eps I)) per Gaussian: conserves the projected splat's integral."""
    I2 = eps * torch.eye(2, dtype=S.dtype, device=S.device)
    a = J @ S @ J.transpose(1, 2) + I2
    b = J @ S2 @ J.transpose(1, 2) + I2
    return torch.sqrt(_det2(a) / _det2(b)).clamp(max=1.0)


def factor_3d(S, S2):
    return torch.sqrt(_det3(S.double()).clamp_min(1e-30) / _det3(S2.double()).clamp_min(1e-30)
                      ).clamp(max=1.0).to(S.dtype)


def rotmat_to_quat(R):
    """(N, 3, 3) proper rotations -> (N, 4) quaternions (w, x, y, z), the convention of _quat_to_rotmat."""
    m = R
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.stack([1 + t, 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
                     1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]], 1)
    k = q.argmax(1)                                                      # Shepperd: divide by the largest component
    s = torch.sqrt(q.gather(1, k[:, None]).clamp_min(1e-12))[:, 0] * 2  # = 4 * |largest component|
    c0, c1, c2 = k == 0, k == 1, k == 2
    w = torch.where(c0, s / 4, torch.where(c1, (m[:, 2, 1] - m[:, 1, 2]) / s, torch.where(
        c2, (m[:, 0, 2] - m[:, 2, 0]) / s, (m[:, 1, 0] - m[:, 0, 1]) / s)))
    x = torch.where(c0, (m[:, 2, 1] - m[:, 1, 2]) / s, torch.where(c1, s / 4, torch.where(
        c2, (m[:, 0, 1] + m[:, 1, 0]) / s, (m[:, 0, 2] + m[:, 2, 0]) / s)))
    y = torch.where(c0, (m[:, 0, 2] - m[:, 2, 0]) / s, torch.where(c1, (m[:, 0, 1] + m[:, 1, 0]) / s, torch.where(
        c2, s / 4, (m[:, 1, 2] + m[:, 2, 1]) / s)))
    z = torch.where(c0, (m[:, 1, 0] - m[:, 0, 1]) / s, torch.where(c1, (m[:, 0, 2] + m[:, 2, 0]) / s, torch.where(
        c2, (m[:, 1, 2] + m[:, 2, 1]) / s, s / 4)))
    out = torch.stack([w, x, y, z], 1)
    return out / out.norm(dim=1, keepdim=True)


def cov_to_quat_scale(S, chunk=1 << 18):
    """Symmetric positive definite (N, 3, 3) -> (quats, scales) with R diag(s^2) R^T = S (the rasteriser's inputs).
    The eigen-decomposition runs on the CPU in chunks: CUDA's batched eigh asks for ~0.5 MB of workspace per 3x3
    matrix (966 GB for 1.9 M Gaussians on a T4); on the CPU 1.9 M take ~5 s."""
    Sc = S.detach().double().cpu()
    evals, evecs = map(torch.cat, zip(*(torch.linalg.eigh(Sc[i:i + chunk]) for i in range(0, len(Sc), chunk)))) \
        if len(Sc) else (Sc.new_zeros(0, 3), Sc.new_zeros(0, 3, 3))
    evals, evecs = evals.to(S.device), evecs.to(S.device)
    flip = _det3(evecs) < 0
    evecs[flip, :, 0] = -evecs[flip, :, 0]
    return rotmat_to_quat(evecs).to(S.dtype), torch.sqrt(evals.clamp_min(1e-20)).to(S.dtype)


# ------------------------------------------------------------------------------------------- per-Gaussian evidence
def base_scales_opacities(params, medium, view):
    """Exactly what gs/render.py feeds the rasteriser (incl. forward scatter if the model has it)."""
    scales, opacities = torch.exp(params["scales"]), torch.sigmoid(params["opacities"])
    scatter = getattr(medium, "scatter", None) if medium is not None else None
    if scatter is not None:
        scales, opacities = scatter.apply(params["means"], params["quats"], scales, opacities, view.viewmat, view.K)
    return scales, opacities


def visibility_weights(params, medium, view, backend):
    """(N,) accumulated blending weight W_ik = sum over pixels of T alpha of every Gaussian in this view: the
    gradient of the summed image with respect to a constant per-Gaussian colour."""
    means = params["means"].detach()
    scales, opacities = base_scales_opacities(params, medium, view)
    col = torch.zeros(1, len(means), 1, device=means.device, requires_grad=True)
    H, W = view.hw
    with torch.enable_grad():
        out, _, _ = rasterize(means, params["quats"].detach(), scales.detach(), opacities.detach(), col,
                              view.viewmat[None], view.K[None], W, H, backend=backend)
        out[..., 0].sum().backward()
    return col.grad[0, :, 0].detach()


@torch.no_grad()
def position_uncertainty(params, medium, train_views, backend, sigma2=0.2, w0=1.0, tau=0.05, sigma_z=0.2):
    """P (N, 3, 3) position covariance from the training views that see each Gaussian, the Mip-style isotropic
    variance (N,), the number of supporting views (N,) and a numerical check of the training-view bound."""
    means = params["means"].detach()
    N, dev = len(means), means.device
    F = torch.zeros(N, 3, 3, dtype=torch.float64, device=dev)
    usum = torch.zeros(N, 3, dtype=torch.float64, device=dev)
    rsum = torch.zeros(N, dtype=torch.float64, device=dev)
    wsum = torch.zeros(N, dtype=torch.float64, device=dev)
    nviews = torch.zeros(N, dtype=torch.int32, device=dev)
    mip_ratio = torch.full((N,), float("inf"), dtype=torch.float64, device=dev)    # min z / f over seeing views
    all_ratio = torch.full((N,), float("inf"), dtype=torch.float64, device=dev)    # ... over all views in front
    full = []                                                                      # (view index, mask w == 1)
    for k, v in enumerate(train_views):
        Wk = visibility_weights(params, medium, v, backend).double()
        J, pc = proj_jacobian(means.double(), v.viewmat.double(), v.K.double())
        sees = Wk > tau
        w = (Wk / w0).clamp(max=1.0) * sees
        F += w[:, None, None] * (J.transpose(1, 2) @ J) / sigma2
        d = means.double() - v.campos.double()
        r = d.norm(dim=1).clamp_min(1e-9)
        usum += w[:, None] * d / r[:, None]
        rsum += w * r
        wsum += w
        nviews += sees.int()
        ratio = pc[:, 2].clamp_min(1e-9) / float(v.K[0, 0])
        mip_ratio = torch.where(sees, torch.minimum(mip_ratio, ratio), mip_ratio)
        all_ratio = torch.where(pc[:, 2] > 0, torch.minimum(all_ratio, ratio), all_ratio)
        full.append((k, sees & (w >= 1.0)))
    seen = wsum > 0
    u = usum / usum.norm(dim=1, keepdim=True).clamp_min(1e-12)
    rbar = torch.where(seen, rsum / wsum.clamp_min(1e-12), torch.zeros_like(rsum))
    lam0 = (u[:, :, None] * u[:, None, :]) / (sigma_z * rbar.clamp_min(1e-6))[:, None, None] ** 2
    eye = torch.eye(3, dtype=torch.float64, device=dev)
    P = torch.zeros(N, 3, 3, dtype=torch.float64, device=dev)
    P[seen] = torch.linalg.inv(F[seen] + lam0[seen] + 1e-12 * eye)
    # never seen by any training view: isotropic, as uncertain as the range prior at the nearest training camera
    if (~seen).any():
        C = torch.stack([v.campos.double() for v in train_views])
        rnear = torch.cdist(means.double()[~seen], C).min(1).values
        P[~seen] = (sigma_z * rnear)[:, None, None] ** 2 * eye
    mip_ratio = torch.where(torch.isfinite(mip_ratio), mip_ratio, all_ratio)
    mip_ratio = torch.where(torch.isfinite(mip_ratio), mip_ratio, torch.ones_like(mip_ratio))
    mipvar = sigma2 * mip_ratio ** 2
    # training-view bound J P J^T <= sigma2 I for views with w = 1 (sampled check)
    worst = 0.0
    for k, mask in full:
        idx = mask.nonzero()[:, 0]
        if len(idx) == 0:
            continue
        idx = idx[torch.randperm(len(idx), device=dev)[:20000]]
        J, _ = proj_jacobian(means.double()[idx], train_views[k].viewmat.double(), train_views[k].K.double())
        worst = max(worst, float(torch.linalg.eigvalsh(J @ P[idx] @ J.transpose(1, 2)).max() / sigma2))
    stats = {"n_gaussians": N, "views_0": int((nviews == 0).sum()), "views_1": int((nviews == 1).sum()),
             "views_2plus": int((nviews >= 2).sum()), "bound_check_max_ratio": worst}
    return P.float(), mipvar.float(), nviews, stats


# ------------------------------------------------------------------------------------------------------ rendering
@torch.no_grad()
def render_variant(params, view, medium, sh_degree, backend, bg_color, mode="base", P=None, mipvar=None, kappa=1.0,
                   cache=None, nviews=None):
    """gs/render.py's render (colour, medium, open-water background) with the covariance / opacity of `mode`.
    `cache` (a dict, optional) keeps each mode's widened covariance and its (quats, scales) across views: they do not
    depend on the camera unless the model has forward scatter (view-dependent scales), in which case it is ignored."""
    means = params["means"]
    v = means - view.campos
    r = v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    d = v / r
    J = torch.clamp_min(eval_sh(sh_degree, sh_coeffs(params), d) + 0.5, 0.0)
    I = medium.compose(J, r, d) if medium is not None else J
    scales, opacities = base_scales_opacities(params, medium, view)
    quats = params["quats"]
    if mode == "cull":
        opacities = torch.where(nviews > 0, opacities, torch.zeros_like(opacities))
    elif mode != "base":
        if cache is not None and getattr(medium, "scatter", None) is not None:
            cache = None
        key = "mip" if mode == "mip" else "P"
        if cache is not None and key in cache:
            S, S2, qs = cache[key]
        else:
            S = gaussian_cov(quats, scales)
            if mode == "mip":
                S2 = S + mipvar[:, None, None] * torch.eye(3, device=S.device)
            else:
                S2 = S + kappa * P
            qs = None
        Jc, _ = proj_jacobian(means, view.viewmat, view.K)
        if mode in ("c2", "fade"):
            fac = factor_2d(Jc, S, S2)
        else:                                   # c2_det3, mip
            fac = factor_3d(S, S2)
        opacities = opacities * fac
        if mode != "fade":
            qs = qs if qs is not None else cov_to_quat_scale(S2)
            quats, scales = qs
        if cache is not None:
            cache[key] = (S, S2, qs if qs is not None else cache.get(key, (None, None, None))[2])
    keep = opacities >= 1.0 / 255.0             # these splats cannot reach any pixel's 1/255 alpha threshold
    H, W = view.hw
    out, alpha, _ = rasterize(means[keep], quats[keep], scales[keep], opacities[keep],
                              torch.cat([I, J], -1)[keep][None], view.viewmat[None], view.K[None], W, H,
                              backend=backend)
    out, alpha = out[0], alpha[0]
    bg = medium.binf(view.dirs) if medium is not None else bg_color.expand(H, W, 3)
    return {"image": out[..., :3] + (1.0 - alpha) * bg, "depth": out[..., 6:7], "alpha": alpha,
            "n_rendered": int(keep.sum())}


# --------------------------------------------------------------------------------------------------- experiment
def _position(view, train_idx):
    if view.idx < min(train_idx):
        return "start"
    if view.idx > max(train_idx):
        return "end"
    return "interior"


@torch.no_grad()
def run(out_dir, tag, device="cuda", backend=None, variants=VARIANTS, sigma2=0.2, kappa=1.0, w0=1.0, tau=0.05,
        sigma_z=0.2, lpips_net="vgg", save_images=False, train_check=True):
    ckpt_path = os.path.join(out_dir, "gs", tag, "ckpt.pt")
    params, medium, bg, cfg = load_model(ckpt_path, device)
    app_state = torch.load(ckpt_path, map_location=device, weights_only=False).get("appearance")
    backend = backend or cfg["backend"]
    data = SceneData(out_dir, device, cfg["downscale"], load_priors=False)
    train = sorted(data.train, key=lambda v: v.train_id)
    train_idx = [v.idx for v in train]
    P, mipvar, nviews, stats = position_uncertainty(params, medium, train, backend, sigma2, w0, tau, sigma_z)
    log(f"[uncertainty] {data.meta['scene']}: {stats}")
    kw = dict(P=P, mipvar=mipvar, kappa=kappa, cache={}, nviews=nviews)   # widened covariances decomposed once per variant
    sh = cfg["sh_degree"]

    test = {}
    img_dir = os.path.join(out_dir, "eval", tag, "uncertainty")
    for v in data.test:
        base = render_variant(params, v, medium, sh, backend, bg, "base")
        seen = seen_mask(v, base["depth"], base["alpha"], train)
        rec = {"frame": v.idx, "position": _position(v, train_idx), "seen_fraction": float(seen.float().mean())}
        tiles = [v.image]
        for mode in variants:
            out = base if mode == "base" else render_variant(params, v, medium, sh, backend, bg, mode, **kw)
            q = quantize(out["image"])
            m = {"psnr": psnr(q, v.image), "ssim": ssim(q, v.image), "lpips": lpips(q, v.image, lpips_net),
                 "n_rendered": out["n_rendered"]}
            if seen.any():
                m["psnr_seen"] = psnr(q, v.image, seen)
            if (~seen).any():
                m["psnr_unseen"] = psnr(q, v.image, ~seen)
            rec[mode] = m
            tiles.append(q)
        if save_images:
            row = torch.cat(tiles, 1).clamp(0, 1).cpu().numpy()
            save_rgb(os.path.join(img_dir, f"{v.stem}_gt_" + "_".join(variants) + ".png"),
                     (row * 255 + 0.5).astype(np.uint8))
        test[v.name] = rec
        log(f"  {v.name} ({rec['position']}, seen {100 * rec['seen_fraction']:.1f}%): " +
            " | ".join(f"{k} {rec[k]['psnr']:.2f}" for k in variants))

    train_res = {}
    if train_check:   # how much each variant changes the training views (with each view's own colour correction)
        dA, db = centred_appearance(app_state, len(train), device)
        for p, v in enumerate(train):
            r = {}
            for mode in variants:
                img = render_variant(params, v, medium, sh, backend, bg, mode, **kw)["image"]
                r[mode] = psnr(quantize(apply_affine(img, dA[p], db[p])), v.image)
            train_res[v.name] = r

    def mean(sel, mode, key):
        vals = [rec[mode][key] for rec in test.values() if sel(rec) and key in rec[mode]]
        return float(np.mean(vals)) if vals else None

    groups = {"all": lambda r: True, "start": lambda r: r["position"] == "start",
              "interior": lambda r: r["position"] == "interior", "end": lambda r: r["position"] == "end"}
    summary = {g: {mode: {key: mean(sel, mode, key) for key in ("psnr", "ssim", "lpips", "psnr_seen", "psnr_unseen")}
                   for mode in variants} for g, sel in groups.items()}
    if train_res:
        summary["train_psnr"] = {mode: float(np.mean([r[mode] for r in train_res.values()])) for mode in variants}
    res = {"scene": data.meta["scene"], "tag": tag,
           "settings": {"sigma2": sigma2, "kappa": kappa, "w0": w0, "tau": tau, "sigma_z": sigma_z, "eps2d": EPS2D,
                        "backend": backend, "colour": "identity (test views), own correction (training views)"},
           "uncertainty_stats": stats, "summary": summary, "test": test, "train": train_res}
    save_json(os.path.join(out_dir, "eval", tag, "uncertainty_splat.json"), res)
    return res
