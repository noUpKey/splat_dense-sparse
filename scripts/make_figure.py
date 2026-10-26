"""Architecture figure of uwsplat, drawn from one trained scene (all panels are real outputs of that run).

python scripts/make_figure.py --out outputs_seathru/Curasao --variant mvs_lpips --save figures/uwsplat_architecture
-> <save>.pdf / .png / .svg

Panels: input frame + SIFT keypoints, dense network depth, MVS class map (verified / network-only / open water),
the trained Gaussians seen from a side viewpoint (restored colour J, no water), restored colour J and depth of the
chosen view, attenuation exp(-beta_D r) and backscatter B_inf(d)(1 - exp(-beta_B r)) at the rendered depth (the model
applies them per Gaussian; per pixel at the expected depth here, for display), the rendered underwater image and the
captured photo, with the training losses.
"""
import argparse
import dataclasses
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uwsplat.eval.evaluate import load_model  # noqa: E402
from uwsplat.gs.data import SceneData  # noqa: E402
from uwsplat.gs.render import render  # noqa: E402

GREEN, RED, BLUE, ORANGE, GREY = "#7cc96b", "#e8706b", "#7ec8e3", "#f3b88f", "#555555"


def u8(x):
    x = x.detach().float().clamp(0, 1).cpu().numpy() if torch.is_tensor(x) else np.clip(x, 0, 1)
    return (x * 255 + 0.5).astype(np.uint8)


def colorize_depth(d, valid):
    inv = np.where(valid, 1.0 / np.maximum(d, 1e-6), 0.0)
    if not valid.any():
        return np.zeros(d.shape + (3,))
    lo, hi = np.percentile(inv[valid], [2, 98])
    c = cm.viridis(np.clip((inv - lo) / max(hi - lo, 1e-8), 0, 1))[..., :3]
    return np.where(valid[..., None], c, 1.0)


def side_view(view, depth, angle_deg=22.0, back=1.35):
    """The same camera turned about a point at the median depth in front of it (shows the 3D structure)."""
    R = view.viewmat[:3, :3]
    C = view.campos
    fwd = R[2]                                      # camera z axis in world coordinates
    d = float(torch.median(depth[depth > 0])) if (depth > 0).any() else 1.0
    c = C + fwd * d
    up = -R[1]
    a = np.deg2rad(angle_deg)
    k = up / up.norm()
    Kx = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], device=C.device)
    Q = torch.eye(3, device=C.device) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx
    C2 = c + back * (Q @ (C - c))
    R2 = R @ Q.T
    vm = torch.eye(4, device=C.device)
    vm[:3, :3], vm[:3, 3] = R2, -R2 @ C2
    return dataclasses.replace(view, viewmat=vm, campos=C2, train_id=-1)


@torch.no_grad()
def panels(out_dir, variant, view_name, device, backend, downscale):
    params, medium, bg, cfg = load_model(os.path.join(out_dir, "gs", variant, "ckpt.pt"), device)
    backend = backend or cfg["backend"]
    data = SceneData(out_dir, device, downscale or cfg["downscale"], load_priors=False)
    views = data.train
    view = next((v for v in views if view_name in (v.stem, v.name)), views[len(views) // 2])
    s = view.image.shape[1] / data.meta["camera"]["width"]
    P = {"name": view.stem, "photo": u8(view.image)}
    pr = next((q for q in (os.path.join(out_dir, g, "priors", view.stem + ".npz") for g in ("mvs", "fusion", "dense"))
               if os.path.exists(q)), None)
    if pr:
        z = np.load(pr)
        P["kp"] = z["kp_uv"] * s
        if "cls" in z:
            cls, dd = z["cls"], z["dense_depth"]
            col = colorize_depth(dd, cls == 2)
            col[cls == 1] = col[cls == 1] * 0.0 + np.array([0.80, 0.80, 0.80])
            col[cls == 0] = np.array([0.60, 0.78, 0.92])
            P["mvs"] = u8(col)
    dn = os.path.join(out_dir, "dense", view.stem + ".npz")
    if os.path.exists(dn):
        z = np.load(dn)
        P["net"] = u8(colorize_depth(z["depth"], z["depth"] > 0))
    o = render(params, view, medium, cfg["sh_degree"], backend, bg_color=bg)
    alpha = o["alpha"][..., 0]
    depth = o["depth"][..., 0]
    P["J"], P["I"], P["gt"] = u8(o["restored"]), u8(o["image"]), u8(view.image)
    P["depth"] = u8(colorize_depth(depth.cpu().numpy(), alpha.cpu().numpy() > 0.5))
    if medium is not None:
        r = (depth * view.ray_norm[..., 0])[..., None]
        att = torch.exp(-medium.beta_d * r) * alpha[..., None]
        binf = medium.binf(view.dirs)
        bs = alpha[..., None] * binf * (1 - torch.exp(-medium.beta_b * r)) + (1 - alpha[..., None]) * binf
        P["att"], P["bs"] = u8(att), u8(bs / bs.max().clamp_min(1e-6))   # backscatter brightened for display
        P["beta"] = (medium.beta_d.tolist(), medium.beta_b.tolist())
    white = torch.ones(3, device=device)
    o2 = render(params, side_view(view, depth), None, cfg["sh_degree"], backend, bg_color=white)
    P["gauss"] = u8(o2["image"])
    P["n_gauss"] = int(params["means"].shape[0])
    return P


def draw(P, save, title=None):
    fig = plt.figure(figsize=(17, 6.2))
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, color, label=None, lw=2.4):
        fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.018",
                                          transform=fig.transFigure, fc="none", ec=color, lw=lw, zorder=0))
        if label:
            fig.text(x + w / 2, y + h - 0.035, label, ha="center", va="top", fontsize=12.5, weight="bold")

    def img(key, x, y, w, h, label=None, below=False, pts=None):
        ax = fig.add_axes([x, y, w, h])
        ax.set_axis_off()
        if key in P:
            ax.imshow(P[key], aspect="auto", interpolation="lanczos")
            if pts is not None and len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=1.2, c="#ff3b3b", lw=0)
        else:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", color=GREY)
        if label:
            fig.text(x + w / 2, y - 0.012 if below else y + h + 0.006, label, ha="center",
                     va="top" if below else "bottom", fontsize=10.5)

    def arrow(x0, y0, x1, y1, color, lw=2.6, style="-|>", ms=16):
        fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle=style,
                                           mutation_scale=ms, color=color, lw=lw, zorder=5))

    def loss(x, y, text, x1=None, y1=None):
        fig.text(x, y, text, ha="center", va="center", fontsize=9.5, weight="bold")
        if x1 is not None:
            arrow(x, y + (0.022 if y1 > y else -0.022), x1, y1, "#cc1f1a", lw=3.2, style="simple", ms=14)

    # 1 multimodal geometry
    box(0.010, 0.17, 0.235, 0.78, GREEN, "Multimodal geometry")
    img("photo", 0.022, 0.585, 0.105, 0.24, "SIFT SfM: poses + points", pts=P.get("kp"))
    img("net", 0.132, 0.585, 0.105, 0.24, "dense network depth")
    img("mvs", 0.050, 0.235, 0.150, 0.27, "MVS check (ZNCC, multi-view)")
    fig.text(0.125, 0.19, "verified  |  network-only  |  open water", ha="center", fontsize=8.5, color=GREY)
    arrow(0.075, 0.58, 0.10, 0.53, GREEN)
    arrow(0.185, 0.58, 0.16, 0.53, GREEN)

    # 2 Gaussians
    box(0.262, 0.17, 0.165, 0.78, GREEN, "3D Gaussians")
    img("gauss", 0.270, 0.30, 0.150, 0.48)
    fig.text(0.345, 0.25, f"init: verified dense cloud\n{P.get('n_gauss', 0):,} Gaussians after training",
             ha="center", va="top", fontsize=9)
    arrow(0.246, 0.56, 0.262, 0.56, GREEN)

    # 3 restored colour + depth
    box(0.445, 0.17, 0.125, 0.78, RED)
    img("J", 0.453, 0.57, 0.110, 0.27, "Restored colour, J")
    img("depth", 0.453, 0.25, 0.110, 0.27, "Depth, z")
    arrow(0.428, 0.56, 0.445, 0.56, GREEN)

    # 4 image formation
    box(0.588, 0.36, 0.230, 0.59, BLUE, "Underwater image formation (per Gaussian)")
    fig.text(0.703, 0.835, r"$I = J\,e^{-\beta^D r} + B^{\infty}(\mathbf{d})\,(1 - e^{-\beta^B r})$",
             ha="center", fontsize=12.5)
    img("att", 0.596, 0.43, 0.105, 0.30, "Attenuation")
    img("bs", 0.706, 0.43, 0.105, 0.30, "Backscatter")
    fig.text(0.703, 0.385, r"empty rays: $B^{\infty}(\mathbf{d})$, water colour by direction", ha="center", fontsize=9)
    arrow(0.571, 0.62, 0.588, 0.62, GREEN)

    # 5 output and photo
    box(0.840, 0.60, 0.150, 0.35, ORANGE)
    img("I", 0.850, 0.64, 0.130, 0.24, r"Underwater render ($\times A_i$)")
    box(0.840, 0.04, 0.150, 0.34, GREY, lw=1.6)
    img("gt", 0.850, 0.08, 0.130, 0.24, "Captured image", below=False)
    arrow(0.819, 0.75, 0.840, 0.75, GREEN)

    # losses
    loss(0.915, 0.47, "L1 + D-SSIM\n+ LPIPS (from 15k)", 0.915, 0.60)
    arrow(0.915, 0.445, 0.915, 0.38, "#cc1f1a", lw=3.2, style="simple", ms=14)
    loss(0.470, 0.085, "verified depth\nprior + smoothness", 0.490, 0.17)
    loss(0.548, 0.085, "grey-world +\nsaturation on J", 0.530, 0.17)
    loss(0.703, 0.24, "backscatter + background\nopacity", 0.703, 0.36)
    fig.text(0.875, 0.50, "+ regulariser\n  on $A_i$", ha="right", fontsize=8.5, color=GREY)
    if title:
        fig.suptitle(title, fontsize=10, y=0.995, color=GREY)
    os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{save}.{ext}", dpi=220 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="scene output folder (with gs/<variant>/ckpt.pt)")
    p.add_argument("--variant", default="mvs_lpips")
    p.add_argument("--view", default="", help="training view stem (default: the middle training view)")
    p.add_argument("--save", default="figures/uwsplat_architecture")
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", default=None)
    p.add_argument("--downscale", type=float, default=0, help="render downscale (default: the training one)")
    a = p.parse_args()
    P = panels(a.out, a.variant, a.view, a.device, a.backend, a.downscale)
    draw(P, a.save)
    print(f"wrote {a.save}.pdf / .png / .svg  (view {P['name']}, {P['n_gauss']:,} Gaussians)")


if __name__ == "__main__":
    main()
