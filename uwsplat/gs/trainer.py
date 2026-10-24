"""Training of the medium-aware Gaussian splatting from the multimodal point cloud."""
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from ..common.io import load_json, log, save_json
from .appearance import Appearance
from .data import SceneData
from .losses import (background_opacity, backscatter_prior, dense_depth_prior, edge_aware_smoothness, esl_smoothness,
                     gray_world, reconstruction, saturation, sparse_depth_prior)
from .medium import WaterMedium
from .model import DENSE, VERIFIED, export_ply, init_params
from .render import render
from .strategy import DensifyStrategy


@dataclass
class TrainConfig:
    iterations: int = 30_000
    geometry: str = "fusion"         # fusion: v1 cloud + priors | mvs: v2 (MVS-verified dense cloud + priors)
    init: str = "fused"              # fused (everything) | sparse | dense | verified (mvs: drop network-only points)
    sh_degree: int = 1
    sh_increase_every: int = 1000
    use_medium: bool = True
    binf_degree: int = 1
    appearance: bool = True
    init_opacity: float = 0.1
    init_opacity_verified: float = 0.0  # > 0: initial opacity of MVS-verified points (0 = init_opacity)
    surfel_ratio: float = 0.0        # > 0: points with a normal start as discs, thickness = ratio x size
    max_init_points: int = 0         # random subsample of the initial cloud (0 = keep all)
    scene_scale: float = 0.0         # > 0: override the 3DGS extent (1.1 x camera radius) that scales the
                                     # position lr and the densify / prune size thresholds; scenes are in
                                     # median-range units, so 0.8 ~ the extent of Isro / Kwaj
    # learning rates: 3DGS defaults, means scaled by the scene extent
    lr_means: float = 1.6e-4
    lr_means_final: float = 1.6e-6
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    lr_medium: float = 5e-3
    lr_appearance: float = 1e-3
    medium_warmup: int = 500
    # loss weights
    lambda_dssim: float = 0.2
    w_dense_depth: float = 0.05      # dense prior, water region
    w_sparse_depth: float = 0.05     # SIFT keypoint depths, wreck
    depth_decay: float = 0.1         # both decay log-linearly to this fraction at the end
    w_smooth: float = 0.01           # edge-aware smoothness of the rendered depth (uwsplat)
    w_esl: float = 0.0               # RUSplatting edge-aware smoothness of the rendered image (their code: 0.05)
    esl_beta: float = 5.0
    esl_detach: bool = True          # False = exactly RUSplatting (depth edges are rewarded, see losses.py)
    w_gray_world: float = 0.01
    w_saturation: float = 0.1
    color_from: int = 3000
    w_backscatter: float = 0.05
    w_bg_opacity: float = 0.01
    bg_thresh: float = 0.08
    w_appearance: float = 0.1
    densify: dict = field(default_factory=dict)  # DensifyStrategy overrides
    backend: str = "gsplat"
    device: str = "cuda"
    downscale: float = 1.0
    eval_every: int = 5000
    log_every: int = 100
    seed: int = 0


@torch.no_grad()
def restored_colors(xyz, rgb, data: SceneData, medium: WaterMedium):
    """Invert the medium at each point's closest observing training camera so that the
    first render already reproduces the observed underwater colours."""
    X = torch.tensor(xyz, dtype=torch.float32, device=data.device)
    best_r = torch.full((len(X),), float("inf"), device=X.device)
    best_d = torch.zeros_like(X)
    for v in data.train:
        pc = X @ v.viewmat[:3, :3].T + v.viewmat[:3, 3]
        z = pc[:, 2]
        uv = (pc[:, :2] / z.clamp_min(1e-6)[:, None]) * torch.diag(v.K)[:2] + v.K[:2, 2]
        H, W = v.hw
        vis = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        dv = X - v.campos
        r = dv.norm(dim=-1)
        better = vis & (r < best_r)
        best_r[better] = r[better]
        best_d[better] = dv[better] / r[better, None]
    miss = ~torch.isfinite(best_r)
    if miss.any():
        C = torch.stack([v.campos for v in data.train])
        dist = (X[miss, None] - C[None]).norm(dim=-1)
        j = dist.argmin(1)
        best_r[miss] = dist.gather(1, j[:, None])[:, 0]
        best_d[miss] = (X[miss] - C[j]) / best_r[miss, None]
    I = torch.tensor(rgb, dtype=torch.float32, device=X.device)
    return medium.restore(I, best_r[:, None], best_d).clamp(0.02, 1.5).cpu().numpy()


def _optimizers(params, cfg: TrainConfig, scene_scale: float):
    lrs = {"means": cfg.lr_means * scene_scale, "scales": cfg.lr_scales, "quats": cfg.lr_quats,
           "opacities": cfg.lr_opacities, "sh0": cfg.lr_sh0, "shN": cfg.lr_shN}
    return {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k], "name": k}], eps=1e-15) for k in params}


@torch.no_grad()
def quick_psnr(params, views, medium, sh_degree, cfg, bg):
    vals = []
    for v in views:
        out = render(params, v, medium, sh_degree, cfg.backend, bg_color=bg)
        mse = ((out["image"].clamp(0, 1) - v.image) ** 2).mean()
        vals.append(float(-10 * torch.log10(mse)))
    return float(np.mean(vals)) if vals else float("nan")


def train(out_dir: str, cfg: TrainConfig, tag: str = "full") -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    gs_dir = os.path.join(out_dir, "gs", tag)
    os.makedirs(gs_dir, exist_ok=True)
    save_json(os.path.join(gs_dir, "config.json"), asdict(cfg))
    dev = cfg.device
    data = SceneData(out_dir, dev, cfg.downscale, geometry=cfg.geometry)
    xyz, rgb, src, nrm = data.points(cfg.init)
    if 0 < cfg.max_init_points < len(xyz):
        keep = np.random.default_rng(cfg.seed).choice(len(xyz), cfg.max_init_points, replace=False)
        xyz, rgb, src, nrm = xyz[keep], rgb[keep], src[keep], nrm[keep]
    scene_scale = cfg.scene_scale if cfg.scene_scale > 0 else data.scene_scale
    log(f"[{tag}] {len(data.train)} train / {len(data.test)} test views at {data.width}x{data.height}; "
        f"init '{cfg.geometry}/{cfg.init}': {int((src == 0).sum())} SIFT + {int((src == DENSE).sum())} network-only"
        f" + {int((src == VERIFIED).sum())} MVS-verified points; scene extent {data.scene_scale:.3f}"
        f"{' -> %.3f (override)' % scene_scale if cfg.scene_scale > 0 else ''}")

    medium, bg = None, None
    mi_path = os.path.join(out_dir, cfg.geometry, "medium_init.json")
    mi = load_json(mi_path if os.path.exists(mi_path) else os.path.join(out_dir, "fusion", "medium_init.json"))
    if cfg.use_medium:
        medium = WaterMedium(mi["binf"], mi["beta_d"], mi["beta_b"], cfg.binf_degree).to(dev)
        rgb = restored_colors(xyz, rgb, data, medium)
        extra_opt = torch.optim.Adam(medium.parameters(), lr=cfg.lr_medium)
    else:
        bg = torch.nn.Parameter(torch.logit(torch.tensor(mi["binf"], dtype=torch.float32, device=dev).clamp(0.01, 0.99)))
        extra_opt = torch.optim.Adam([bg], lr=cfg.lr_medium)
    opacity = np.full(len(xyz), cfg.init_opacity)
    if cfg.init_opacity_verified > 0:
        opacity[src == VERIFIED] = cfg.init_opacity_verified
    params = init_params(xyz, rgb, opacity, cfg.sh_degree, dev, nrm, cfg.surfel_ratio)
    optimizers = _optimizers(params, cfg, scene_scale)
    sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=(cfg.lr_means_final / cfg.lr_means) ** (1.0 / cfg.iterations))
    appearance = Appearance(len(data.train)).to(dev) if cfg.appearance else None
    app_opt = torch.optim.Adam(appearance.parameters(), lr=cfg.lr_appearance) if appearance is not None else None
    strategy = DensifyStrategy(**cfg.densify)
    state = strategy.initialize_state(scene_scale, torch.tensor(src, device=dev))

    log_f = open(os.path.join(gs_dir, "train_log.jsonl"), "w")
    order, t0, ema = [], time.time(), None
    for step in range(cfg.iterations):
        if not order:
            order = list(range(len(data.train)))
            random.shuffle(order)
        view = data.train[order.pop()]
        sh_deg = min(step // cfg.sh_increase_every, cfg.sh_degree)
        bg_rgb = torch.sigmoid(bg) if bg is not None else None
        out = render(params, view, medium, sh_deg, cfg.backend, strategy.absgrad, appearance, bg_rgb)
        strategy.step_pre_backward(out["meta"])
        gt, depth, alpha = view.image, out["depth"], out["alpha"]

        L = {"rgb": reconstruction(out["image"], gt, cfg.lambda_dssim)}
        decay = cfg.depth_decay ** (step / cfg.iterations)
        if cfg.w_dense_depth > 0 and view.prior_depth is not None:
            L["dense"] = cfg.w_dense_depth * decay * dense_depth_prior(depth, alpha, view.prior_depth, view.prior_w)
        if cfg.w_sparse_depth > 0 and view.kp_uv is not None and len(view.kp_uv):
            L["sparse"] = cfg.w_sparse_depth * decay * sparse_depth_prior(depth, alpha, view.kp_uv, view.kp_z)
        if cfg.w_smooth > 0:
            L["smooth"] = cfg.w_smooth * edge_aware_smoothness(depth, alpha, gt)
        if cfg.w_esl > 0:
            L["esl"] = cfg.w_esl * esl_smoothness(out["image"], depth, alpha, cfg.esl_beta, cfg.esl_detach)
        if medium is not None:
            if step >= cfg.color_from:
                L["gray"] = cfg.w_gray_world * gray_world(out["restored"], alpha)
                L["sat"] = cfg.w_saturation * saturation(out["restored"])
            if step >= cfg.medium_warmup:
                L["backscatter"] = cfg.w_backscatter * backscatter_prior(gt, depth, alpha, view.ray_norm,
                                                                         out["background"], medium.beta_b)
                L["bg_opacity"] = cfg.w_bg_opacity * background_opacity(alpha, gt, out["background"], cfg.bg_thresh)
        if appearance is not None:
            L["appearance"] = cfg.w_appearance * appearance.regularizer()
        loss = sum(L.values())
        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if step >= cfg.medium_warmup:
            extra_opt.step()
        extra_opt.zero_grad(set_to_none=True)
        if app_opt is not None:
            app_opt.step()
            app_opt.zero_grad(set_to_none=True)
        sched.step()
        dstats = strategy.step_post_backward(params, optimizers, state, step, out["meta"])

        with torch.no_grad():  # running train PSNR kept on the device: no per-step GPU sync
            psnr = -10 * torch.log10(((out["image"].detach().clamp(0, 1) - gt) ** 2).mean())
            ema = psnr if ema is None else 0.95 * ema + 0.05 * psnr
        if step % cfg.log_every == 0 or step == cfg.iterations - 1:
            rec = {"step": step, "loss": loss.item(), "train_psnr_ema": ema.item(), "n_gaussians": len(params["means"]),
                   **{k: v.item() for k, v in L.items()}, **dstats}
            if medium is not None:
                rec["beta_d"] = [round(x, 4) for x in medium.beta_d.tolist()]
                rec["beta_b"] = [round(x, 4) for x in medium.beta_b.tolist()]
            # step 0 included: test PSNR of the initial cloud alone (v1 vs v2 initialisation)
            if cfg.eval_every and (step % cfg.eval_every == 0 or step == cfg.iterations - 1) and data.test:
                rec["test_psnr"] = quick_psnr(params, data.test, medium, sh_deg, cfg, bg_rgb)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if step % (cfg.log_every * 10) == 0 or "test_psnr" in rec:
                it_s = (step + 1) / (time.time() - t0)
                log(f"[{tag}] step {step:6d} loss {rec['loss']:.4f} train PSNR {rec['train_psnr_ema']:.2f} "
                    f"{'test PSNR %.2f ' % rec['test_psnr'] if 'test_psnr' in rec else ''}"
                    f"#GS {len(params['means'])} ({it_s:.1f} it/s)")
    log_f.close()

    ckpt = {"params": {k: v.detach().cpu() for k, v in params.items()}, "cfg": asdict(cfg),
            "medium": medium.state_dict() if medium is not None else None,
            "bg": bg.detach().cpu() if bg is not None else None,
            "appearance": appearance.state_dict() if appearance is not None else None,
            "modality": state["modality"].cpu(), "scene_scale": scene_scale, "scene_scale_auto": data.scene_scale,
            "train_time_s": time.time() - t0}
    torch.save(ckpt, os.path.join(gs_dir, "ckpt.pt"))
    export_ply(params, os.path.join(gs_dir, "point_cloud.ply"))
    final = {"n_gaussians": len(params["means"]), "n_from_dense": int((state["modality"] == DENSE).sum()),
             "n_from_verified": int((state["modality"] == VERIFIED).sum()),
             "n_init": {"sift": int((src == 0).sum()), "network": int((src == DENSE).sum()),
                        "verified": int((src == VERIFIED).sum())},
             "scene_scale": scene_scale, "scene_scale_auto": data.scene_scale,
             "train_time_s": round(time.time() - t0, 1), "medium": medium.summary() if medium is not None else None}
    save_json(os.path.join(gs_dir, "final.json"), final)
    log(f"[{tag}] done in {final['train_time_s']}s, {final['n_gaussians']} Gaussians")
    return final
