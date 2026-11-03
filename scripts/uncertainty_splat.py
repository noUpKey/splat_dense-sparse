"""Geometric-uncertainty splatting (C2) on a trained uwsplat model: no training, re-render and score.

python scripts/uncertainty_splat.py --scene_out outputs/Tokai --tag mvs_lpips [--variants base,c2,fade,c2_det3,mip,cull]

Writes <scene_out>/eval/<tag>/uncertainty_splat.json (per test view: PSNR / SSIM / LPIPS, PSNR on seen and
never-seen pixels, sequence position; per training view: PSNR with its own colour correction) and, with
--save_images, side-by-side PNGs (GT | variants) in <scene_out>/eval/<tag>/uncertainty/.
See uwsplat/eval/uncertainty.py for the method and its pre-registered settings (sigma2 = 0.2 px^2, kappa = 1,
W0 = 1, tau = 0.05, sigma_z = 0.2).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uwsplat.eval.uncertainty import VARIANTS, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene_out", required=True, help="uwsplat output folder of one scene (sfm/, images/, gs/<tag>/)")
    ap.add_argument("--tag", default="mvs_lpips")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--sigma2", type=float, default=0.2, help="pixel variance of an observation (px^2)")
    ap.add_argument("--kappa", type=float, default=1.0, help="scale of P added to the covariance")
    ap.add_argument("--w0", type=float, default=1.0, help="blending weight (pixels) at which a view counts fully")
    ap.add_argument("--tau", type=float, default=0.05, help="minimum blending weight for a view to count at all")
    ap.add_argument("--sigma_z", type=float, default=0.2, help="range prior: std as a fraction of the mean range")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default=None, help="gsplat (default from the checkpoint) or torch (CPU, tiny scenes)")
    ap.add_argument("--save_images", action="store_true")
    ap.add_argument("--no_train_check", action="store_true", help="skip re-rendering the training views")
    a = ap.parse_args()
    variants = tuple(x for x in a.variants.split(",") if x)
    bad = [x for x in variants if x not in VARIANTS]
    if bad:
        sys.exit(f"unknown variants {bad}; choose from {VARIANTS}")
    if "base" not in variants:
        variants = ("base",) + variants
    run(a.scene_out, a.tag, a.device, a.backend, variants, a.sigma2, a.kappa, a.w0, a.tau, a.sigma_z,
        save_images=a.save_images, train_check=not a.no_train_check)


if __name__ == "__main__":
    main()
