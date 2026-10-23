"""Render the held-out test views and score them (PSNR / SSIM / LPIPS, plus region-wise PSNR)."""
import os

import matplotlib.cm as cm
import numpy as np
import torch

from ..common.io import log, save_json, save_rgb
from ..gs.data import SceneData
from ..gs.medium import WaterMedium
from ..gs.render import render
from .metrics import lpips, psnr, quantize, ssim


def _u8(x):
    return (x.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def _depth_viz(depth, alpha):
    d = depth[..., 0].cpu().numpy()
    m = alpha[..., 0].cpu().numpy() > 0.5
    if not m.any():
        return np.zeros(d.shape + (3,), np.uint8)
    inv = np.where(m, 1.0 / np.maximum(d, 1e-6), 0)
    lo, hi = np.percentile(inv[m], [2, 98])
    c = cm.turbo(np.clip((inv - lo) / max(hi - lo, 1e-8), 0, 1))[..., :3]
    return (c * m[..., None] * 255).astype(np.uint8)


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device)) for k, v in ckpt["params"].items()})
    cfg = ckpt["cfg"]
    medium, bg = None, None
    if ckpt["medium"] is not None:
        binf_deg = int(round(np.sqrt(ckpt["medium"]["binf_sh"].shape[0]))) - 1
        medium = WaterMedium(np.full(3, 0.5), np.full(3, 0.5), np.full(3, 0.5), binf_deg).to(device)
        medium.load_state_dict(ckpt["medium"])
    if ckpt["bg"] is not None:
        bg = torch.sigmoid(ckpt["bg"].to(device))
    return params, medium, bg, cfg


@torch.no_grad()
def evaluate(out_dir: str, tag: str = "full", device: str = "cuda", backend: str = None, save_images: bool = True,
             lpips_net: str = "vgg") -> dict:
    params, medium, bg, cfg = load_model(os.path.join(out_dir, "gs", tag, "ckpt.pt"), device)
    backend = backend or cfg["backend"]
    data = SceneData(out_dir, device, cfg["downscale"], load_priors=False)
    ev_dir = os.path.join(out_dir, "eval", tag)
    per_view = {}
    missing = [im["name"] for im in data.meta["images"] if im["split"] == "test" and not im["registered"]]
    for v in data.test:
        out = render(params, v, medium, cfg["sh_degree"], backend, bg_color=bg)
        pred, gt = quantize(out["image"]), v.image
        m = {"psnr": psnr(pred, gt), "ssim": ssim(pred, gt), "lpips": lpips(pred, gt, lpips_net)}
        if v.wreck is not None and v.wreck.any() and (~v.wreck).any():
            m["psnr_wreck"] = psnr(pred, gt, v.wreck)
            m["psnr_water"] = psnr(pred, gt, ~v.wreck)
            m["wreck_fraction"] = float(v.wreck.float().mean())
        per_view[v.name] = m
        if save_images:
            tiles = [_u8(gt), _u8(pred), _u8(out["restored"]), _depth_viz(out["depth"], out["alpha"])]
            for name, tile in zip(("gt", "renders", "restored", "depth"), tiles):
                save_rgb(os.path.join(ev_dir, name, v.stem + ".png"), tile)
            save_rgb(os.path.join(ev_dir, "montage", v.stem + ".jpg"), np.concatenate(tiles, 1))
    keys = sorted({k for m in per_view.values() for k in m})
    mean = {k: float(np.mean([m[k] for m in per_view.values() if k in m])) for k in keys}
    train_psnr = float(np.mean([psnr(quantize(render(params, v, medium, cfg["sh_degree"], backend, bg_color=bg)["image"]),
                                     v.image) for v in data.train]))
    res = {"scene": data.meta["scene"], "tag": tag, "mean": mean, "per_view": per_view, "train_psnr": train_psnr,
           "missing_test_views": missing, "n_gaussians": int(len(params["means"])), "lpips_net": lpips_net}
    save_json(os.path.join(ev_dir, "metrics.json"), res)
    log(f"[{tag}] {data.meta['scene']}: PSNR {mean['psnr']:.2f}  SSIM {mean['ssim']:.4f}  LPIPS {mean['lpips']:.4f}"
        f"  (train PSNR {train_psnr:.2f}, {len(per_view)} test views)")
    return res
