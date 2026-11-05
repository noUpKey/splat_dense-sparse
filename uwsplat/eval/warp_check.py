"""Training-free check of the water-aware image-based residual idea (project note `water-aware-ibr-residual`).

Question: before training any network that adds a residual from neighbouring training photos to the splat render,
does (1) a physically correct transfer of a neighbour's pixel into the test view beat a plain camera-colour transfer,
and (2) does the transferred photo carry detail that the render lacks? Everything here uses a frozen checkpoint, the
training photos and the model's own geometry; no test pixel is used except for scoring.

For a test view t, each surface pixel (rendered alpha > alpha_min) is lifted to 3D with the rendered depth, projected
into source training views s (training frames only, never an interpolated or test frame), and kept where the source
sees it (inside the image, source alpha > alpha_min, depth agrees within occ_tol: an occlusion test). The source photo
I_s is sampled there (bilinear). With the trained per-frame colour correction C_k(x) = (Id + dA_k) x + db_k and the
medium I = J e^(-bD r) + B(d) (1 - e^(-bB r)) (r = range along the ray, d = world ray direction):

    model-space source   M_s = C_s^-1(I_s)
    (a) naive            I_s                                             photo copied as is (IBGS without exposure fix)
    (b) camera           C_t(M_s)                                        source camera removed, target camera applied
    (c) camera + water   C_t( (M_s - B(d_s)(1 - e^(-bB r_s))) e^(-bD (r_t - r_s)) + B(d_t)(1 - e^(-bB r_t)) )
    (d) render           C_t(R_t)                                        the current model (= `neighbors` protocol)
    (e) render + residual, water-scaled    C_t( R_t + (M_s - R_s(u_s)) e^(-bD (r_t - r_s)) )
    (f) render + residual, unscaled        C_t( R_t + (M_s - R_s(u_s)) )

R_s(u_s) is the source's raw render at the projected point, so (e)/(f) are what a residual network would start from
with no learning. C_t is the target frame's colour correction predicted without test pixels (`neighbors`: linear in
time between the training frames before and after, as in heldout.py; or `identity` for photo collections such as
SeaThru-NeRF, where file order carries no camera state). Sources are combined with their weights; a pixel is scored
only if every source sees it ("common" mask), so all variants are compared on the same pixels.

Scores per test view, on the common mask: PSNR (8-bit quantised, as evaluate()) and high-frequency correlation
(Pearson correlation of the difference-of-Gaussians band of prediction and photo; source noise is uncorrelated with
the test photo's noise, so this measures shared detail where PSNR also charges the source's noise). Full-image
PSNR / SSIM / LPIPS of the composite (variant inside the mask, render outside) show what the image would score.
Also logged: how much the range changes between source and target (|r_t - r_s| / r_t) and how far (c) moves from (b)
in 8-bit levels, i.e. how large the water term actually is on this data.

Pre-registered rule (fixed before any real run, project note `water-aware-ibr-residual`):
  physics: (c) beats (b) on masked PSNR by > 0.1 dB in >= 3 of 4 scenes;
  detail:  (c) beats (d) on masked PSNR in >= 3 of 4 scenes. Secondary, reported as such: (c) beats (d) on
           high-frequency correlation in >= 3 of 4 scenes (PSNR charges warped source noise that a learned
           residual would remove; if only this one passes, the detail claim is weaker and says so).
  Build the residual network only if physics and at least one detail criterion pass.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from ..common.io import log, save_json, save_rgb
from ..gs.data import SceneData
from ..gs.render import render
from .evaluate import load_model
from .heldout import apply_affine, blend, centred_appearance, temporal_neighbors
from .metrics import lpips, psnr, quantize, ssim

VARIANTS = ("naive", "camera", "camera_water", "render", "render_res_water", "render_res")


# ------------------------------------------------------------------------------------------------- geometry
def surface_points(view, depth: torch.Tensor) -> torch.Tensor:
    """(H, W, 3) world points of the rendered surface (z-depth from the rasteriser's expected depth)."""
    H, W = view.hw
    dev = depth.device
    v, u = torch.meshgrid(torch.arange(H, device=dev) + 0.5, torch.arange(W, device=dev) + 0.5, indexing="ij")
    K = view.K
    rays = torch.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], torch.ones_like(u)], -1)
    R, t = view.viewmat[:3, :3], view.viewmat[:3, 3]
    return (rays * depth[..., :1] - t) @ R        # camera -> world for row vectors: R^T (x - t)


def project(view, X: torch.Tensor):
    """World points (..., 3) -> pixel coords u, v and z-depth in `view`."""
    P = X @ view.viewmat[:3, :3].T + view.viewmat[:3, 3]
    z = P[..., 2]
    zs = torch.where(z > 1e-6, z, torch.ones_like(z))
    K = view.K
    return K[0, 0] * P[..., 0] / zs + K[0, 2], K[1, 1] * P[..., 1] / zs + K[1, 2], z


def sample(img: torch.Tensor, u: torch.Tensor, v: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """img (H, W, C) at pixel coords (u, v) (pixel centres at +0.5), zeros outside."""
    H, W = img.shape[:2]
    g = torch.stack([u / W * 2 - 1, v / H * 2 - 1], -1)[None]
    out = F.grid_sample(img.permute(2, 0, 1)[None], g, mode=mode, padding_mode="zeros", align_corners=False)
    return out[0].permute(1, 2, 0)


def inverse_affine(img: torch.Tensor, dA: torch.Tensor, db: torch.Tensor) -> torch.Tensor:
    """Inverse of apply_affine: (img - db) (Id + dA)^-T."""
    M = torch.eye(3, device=img.device) + dA
    return (img - db) @ torch.linalg.inv(M).T


# --------------------------------------------------------------------------------------------------- scores
def _dog(x: torch.Tensor, s1: float = 1.0, s2: float = 2.5) -> torch.Tensor:
    """Difference-of-Gaussians band of a (H, W, 3) image (separable blur, reflect padding)."""
    def blur(img, s):
        r = int(3 * s + 0.5)
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=img.device, dtype=img.dtype) / s) ** 2)
        k = (k / k.sum())
        y = img.permute(2, 0, 1)[:, None]                                    # (3, 1, H, W)
        y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
        y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
        return y[:, 0].permute(1, 2, 0)
    return blur(x, s1) - blur(x, s2)


def erode(mask: torch.Tensor, r: int) -> torch.Tensor:
    """Binary erosion of a (H, W) mask by a (2r+1)^2 square."""
    m = (~mask).float()[None, None]
    return ~(F.max_pool2d(m, 2 * r + 1, stride=1, padding=r)[0, 0] > 0)


def hf_corr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, s2: float = 2.5) -> float:
    """Correlation of the DoG bands on the mask eroded by the filter radius, so the band never sees the mask edge.
    `pred` should be a full image (variant inside the mask, render outside), not zeros outside."""
    inner = erode(mask, int(3 * s2 + 0.5) + 1)
    if inner.sum() < 50:
        return float("nan")
    a, b = _dog(pred)[inner].reshape(-1), _dog(gt)[inner].reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def _full_scores(pred, gt, lpips_net):
    q = quantize(pred)
    return {"psnr": psnr(q, gt), "ssim": ssim(q, gt), "lpips": lpips(q, gt, lpips_net)}


# ------------------------------------------------------------------------------------------------- transfer
def transfer(view, X, r_t, d_t, src, src_img, src_render, src_depth, src_alpha, dA_s, db_s, medium,
             alpha_min=0.95, occ_tol=0.03):
    """Model-space candidates from one source view; returns (valid (H, W), dict of (H, W, 3), extras)."""
    u, v, z = project(src, X)
    Hs, Ws = src.hw
    inside = (z > 1e-6) & (u >= 0) & (u < Ws) & (v >= 0) & (v < Hs)
    z_s = sample(src_depth, u, v, "nearest")[..., 0]
    a_s = sample(src_alpha, u, v, "nearest")[..., 0]
    valid = inside & (a_s > alpha_min) & ((z - z_s).abs() <= occ_tol * z_s.clamp_min(1e-6))
    I_s = sample(src_img, u, v)
    R_s = sample(src_render, u, v)
    M_s = inverse_affine(I_s, dA_s, db_s)
    vs = X - src.campos
    r_s = vs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    d_s = vs / r_s
    bd, bb = medium.beta_d, medium.beta_b
    ratio = torch.exp(-bd * (r_t - r_s))
    bs_s = medium.binf(d_s) * (1 - torch.exp(-bb * r_s))
    bs_t = medium.binf(d_t) * (1 - torch.exp(-bb * r_t))
    cand = {"naive": I_s, "camera": M_s, "camera_water": (M_s - bs_s) * ratio + bs_t,
            "res_water": (M_s - R_s) * ratio, "res": M_s - R_s}
    return valid, cand, {"r_s": r_s[..., 0], "ratio": ratio}


@torch.no_grad()
def run(out_dir: str, tag: str = "mvs_lpips", device: str = "cuda", backend: str = None, sources: str = "temporal",
        k: int = 2, target_colour: str = "neighbors", alpha_min: float = 0.95, occ_tol: float = 0.03,
        lpips_net: str = "vgg", save_images: bool = True) -> dict:
    if sources not in ("temporal", "overlap"):
        raise ValueError("sources: temporal | overlap")
    if target_colour not in ("neighbors", "identity"):
        raise ValueError("target_colour: neighbors | identity")
    ckpt_path = os.path.join(out_dir, "gs", tag, "ckpt.pt")
    params, medium, bg, cfg = load_model(ckpt_path, device)
    if medium is None:
        raise ValueError("the water transfer needs a model with a medium")
    app_state = torch.load(ckpt_path, map_location=device, weights_only=False).get("appearance")
    backend = backend or cfg["backend"]
    data = SceneData(out_dir, device, cfg["downscale"], load_priors=False)
    train = sorted(data.train, key=lambda v: v.train_id)
    train_idx = [v.idx for v in train]
    dA, db = centred_appearance(app_state, len(train), device)
    zero = (torch.zeros(3, 3, device=device), torch.zeros(3, device=device))
    rnd = lambda v: render(params, v, medium, cfg["sh_degree"], backend, bg_color=bg)

    src_cache = {}

    def source(p):
        if p not in src_cache:
            o = rnd(train[p])
            src_cache[p] = (o["image"], o["depth"], o["alpha"])
        return src_cache[p]

    ev_dir = os.path.join(out_dir, "eval", tag, "warp_check")
    per_view = {}
    for view in data.test:
        out = rnd(view)
        R_t, alpha_t = out["image"], out["alpha"]
        surf = alpha_t[..., 0] > alpha_min
        X = surface_points(view, out["depth"])
        vt = X - view.campos
        r_t = vt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        d_t = vt / r_t

        # sources: temporal neighbours (video) or the k training views that see most of the surface (photo sets)
        if sources == "temporal":
            ws = temporal_neighbors(view.idx, train_idx)
        else:
            cover = []
            for p, s in enumerate(train):
                img_s, z_s, a_s = source(p)
                u, v, z = project(s, X)
                Hs, Ws = s.hw
                ins = surf & (z > 1e-6) & (u >= 0) & (u < Ws) & (v >= 0) & (v < Hs)
                zz = sample(z_s, u, v, "nearest")[..., 0]
                ok = ins & ((z - zz).abs() <= occ_tol * zz.clamp_min(1e-6))
                cover.append(float(ok.float().mean()))
            best = sorted(range(len(train)), key=lambda p: -cover[p])[:k]
            ws = [(p, 1.0 / len(best)) for p in best]
        if not ws:
            continue
        A_t, b_t = blend(temporal_neighbors(view.idx, train_idx), dA, db) \
            if target_colour == "neighbors" and temporal_neighbors(view.idx, train_idx) else zero

        mask = surf.clone()
        acc = {k_: torch.zeros_like(R_t) for k_ in ("naive", "camera", "camera_water", "res_water", "res")}
        wsum = sum(w for _, w in ws)
        rel_dr, ratios = [], []
        for p, w in ws:
            img_s = train[p].image
            R_s, z_s, a_s = source(p)
            valid, cand, ex = transfer(view, X, r_t, d_t, train[p], img_s, R_s, z_s, a_s, dA[p], db[p], medium,
                                       alpha_min, occ_tol)
            mask &= valid
            for key in acc:
                acc[key] += (w / wsum) * cand[key]
            rel_dr.append(((r_t[..., 0] - ex["r_s"]).abs() / r_t[..., 0]))
            ratios.append(ex["ratio"])
        r = {"frame": view.idx, "sources": [[train[p].stem, round(float(w), 3)] for p, w in ws],
             "surface_fraction": float(surf.float().mean()), "mask_fraction": float(mask.float().mean())}
        if mask.sum() < 100:
            r["skipped"] = "fewer than 100 commonly seen surface pixels"
            per_view[view.name] = r
            continue
        C_t = lambda x: apply_affine(x, A_t, b_t)
        preds = {"naive": acc["naive"], "camera": C_t(acc["camera"]), "camera_water": C_t(acc["camera_water"]),
                 "render": C_t(R_t), "render_res_water": C_t(R_t + acc["res_water"]),
                 "render_res": C_t(R_t + acc["res"])}
        gt = view.image
        comp = {k_: torch.where(mask[..., None], x, preds["render"]) for k_, x in preds.items()}
        r["masked"] = {k_: {"psnr": psnr(quantize(x), gt, mask), "hf_corr": hf_corr(quantize(comp[k_]), gt, mask)}
                       for k_, x in preds.items()}
        r["full_composite"] = {k_: _full_scores(x, gt, lpips_net) for k_, x in comp.items()}
        dr = torch.stack(rel_dr)[:, mask]
        ratio = torch.stack(ratios)[:, mask]
        r["water_term"] = {
            "rel_range_change_median": float(dr.median()), "rel_range_change_p90": float(torch.quantile(dr, 0.9)),
            "attenuation_ratio_dev_median": float((ratio - 1).abs().median()),
            "water_minus_camera_8bit_mean": float((preds["camera_water"] - preds["camera"])[mask].abs().mean() * 255)}
        per_view[view.name] = r
        if save_images:
            u8 = lambda x: (x.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
            m3 = mask[..., None].float().expand(-1, -1, 3)
            tiles = [u8(gt), u8(preds["render"]), u8(torch.where(mask[..., None], preds["camera_water"], gt * 0.3)),
                     u8(torch.where(mask[..., None], preds["render_res_water"], preds["render"])), u8(m3)]
            save_rgb(os.path.join(ev_dir, view.stem + ".jpg"), np.concatenate(tiles, 1))

    scored = [r for r in per_view.values() if "masked" in r]
    summary = {}
    if scored:
        summary = {
            "masked": {k_: {m: float(np.nanmean([r["masked"][k_][m] for r in scored])) for m in ("psnr", "hf_corr")}
                       for k_ in VARIANTS},
            "full_composite": {k_: {m: float(np.mean([r["full_composite"][k_][m] for r in scored]))
                                    for m in ("psnr", "ssim", "lpips")} for k_ in VARIANTS},
            "mask_fraction": float(np.mean([r["mask_fraction"] for r in scored])),
            "rel_range_change_median": float(np.mean([r["water_term"]["rel_range_change_median"] for r in scored])),
            "water_minus_camera_8bit_mean": float(np.mean([r["water_term"]["water_minus_camera_8bit_mean"]
                                                           for r in scored])),
            "n_views": len(scored)}
        mk = summary["masked"]
        summary["physics_delta_db"] = mk["camera_water"]["psnr"] - mk["camera"]["psnr"]
        summary["detail_delta_db"] = mk["camera_water"]["psnr"] - mk["render"]["psnr"]
        summary["detail_delta_hf"] = mk["camera_water"]["hf_corr"] - mk["render"]["hf_corr"]
    res = {"scene": data.meta["scene"], "tag": tag, "sources": sources, "k": k, "target_colour": target_colour,
           "alpha_min": alpha_min, "occ_tol": occ_tol, "lpips_net": lpips_net, "summary": summary,
           "per_view": per_view}
    save_json(os.path.join(out_dir, "eval", tag, "warp_check.json"), res)
    if summary:
        mk = summary["masked"]
        log(f"[warp check] {data.meta['scene']} masked PSNR: " + " | ".join(f"{k_} {mk[k_]['psnr']:.2f}"
                                                                           for k_ in VARIANTS))
        log(f"[warp check] {data.meta['scene']} masked HF corr: " + " | ".join(f"{k_} {mk[k_]['hf_corr']:.3f}"
                                                                              for k_ in VARIANTS))
        log(f"[warp check] {data.meta['scene']} physics (c-b) {summary['physics_delta_db']:+.2f} dB, detail (c-d) "
            f"{summary['detail_delta_db']:+.2f} dB / {summary['detail_delta_hf']:+.3f} HF corr; mask "
            f"{100 * summary['mask_fraction']:.0f} %, range change {100 * summary['rel_range_change_median']:.1f} %, "
            f"water term {summary['water_minus_camera_8bit_mean']:.2f} levels")
    else:
        log(f"[warp check] {data.meta['scene']}: no test view had enough commonly seen pixels")
    return res


def decide(summaries: dict) -> dict:
    """Apply the pre-registered rule to {scene: summary}."""
    s = {k: v for k, v in summaries.items() if v}
    n = len(s)
    need = 3 if n >= 4 else max(1, n - 1)
    phys = sum(v["physics_delta_db"] > 0.1 for v in s.values())
    det_psnr = sum(v["detail_delta_db"] > 0 for v in s.values())
    det_hf = sum(v["detail_delta_hf"] > 0 for v in s.values())
    return {"n_scenes": n, "needed": need, "physics_pass": phys, "detail_psnr_pass": det_psnr,
            "detail_hf_pass": det_hf, "physics": phys >= need, "detail_psnr": det_psnr >= need,
            "detail_hf": det_hf >= need,
            "build_network": phys >= need and (det_psnr >= need or det_hf >= need)}
