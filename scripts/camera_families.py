"""E2: refit per-frame camera colour families on a trained uwsplat model and predict held-out frames. No training.

python scripts/camera_families.py --scene_out outputs/Tokai --tag mvs_lpips [--families A,B,C,D,Do,E] [--stride 2]

Writes <scene_out>/eval/<tag>/camera_families.json (test PSNR / SSIM / LPIPS per family with neighbour-interpolated
parameters; training leave-one-out PSNR; fitted offsets, gains and shared M). See uwsplat/eval/camera_families.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uwsplat.eval.camera_families import FAMILIES, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene_out", required=True)
    ap.add_argument("--tag", default="mvs_lpips")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--stride", type=int, default=2, help="fit on every stride-th pixel in each direction")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default=None)
    a = ap.parse_args()
    fams = tuple(f for f in a.families.split(",") if f)
    bad = [f for f in fams if f not in FAMILIES]
    if bad:
        sys.exit(f"unknown families {bad}; choose from {FAMILIES}")
    run(a.scene_out, a.tag, a.device, a.backend, fams, a.stride)


if __name__ == "__main__":
    main()
