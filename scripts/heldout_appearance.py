"""Re-score trained models on the held-out views with other test-view colour corrections (no retraining).

python scripts/heldout_appearance.py --out outputs_v2 --scenes Tokai,Cormoran,Isro,Kwaj --variants full,mvs,mvs_lpips
python scripts/heldout_appearance.py --out outputs_v2 --scenes ... --variants ... --report   # print only
python scripts/heldout_appearance.py ... --per-view mvs_lpips    # whose test views are listed one by one

Modes (uwsplat/eval/heldout.py): identity (= evaluate(), the training average), neighbors (interpolated in time
from the training frames before and after the view), nearest_pose (nearest training camera), oracle (affine fitted
to the test image: a bound, not a result). Writes <out>/<scene>/eval/<variant>/heldout_appearance.json; runs that
already have it are skipped unless --force. Needs the trained checkpoint (gs/<variant>/ckpt.pt).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from uwsplat.eval.heldout import MODES  # noqa: E402

ALL = MODES + ("oracle",)


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def _short(name):
    return os.path.splitext(name)[0][-8:]


def report(out: str, scenes: list, variants: list, per_view: str = None):
    rows = {v: [] for v in variants}
    print(f"{'scene':9} {'variant':16} " + " ".join(f"{m:>21}" for m in ALL) + "   train leave-one-out (PSNR)")
    print(f"{'':9} {'':16} " + " ".join(f"{'PSNR / SSIM / LPIPS':>21}" for _ in ALL)
          + "   identity neighbors nearest own")
    for scene in scenes:
        for v in variants:
            r = _load(os.path.join(out, scene, "eval", v, "heldout_appearance.json"))
            if r is None:
                continue
            mt, tr = r["test"]["mean"], r["train_leave_one_out"]["mean"]
            rows[v].append(mt)
            cells = " ".join(f"{mt[m]['psnr']:6.2f} /{mt[m]['ssim']:6.3f} /{mt[m]['lpips']:6.3f}" for m in ALL)
            print(f"{scene:9} {v:16} {cells}   {tr['identity']:8.2f} {tr['neighbors']:9.2f} {tr['nearest_pose']:7.2f}"
                  f" {tr['own']:5.2f}" + ("" if r["has_appearance"] else "   (no colour model)"))
    print()
    for v, ms in rows.items():
        if not ms:
            continue
        mean = {m: {k: float(np.mean([x[m].get(k, np.nan) for x in ms])) for k in ("psnr", "ssim", "lpips", "psnr_seen")}
                for m in ALL}
        base = mean["identity"]
        print(f"{v:16} mean over {len(ms)} scene(s): " + " | ".join(
            f"{m} {mean[m]['psnr']:.2f} / {mean[m]['ssim']:.4f} / {mean[m]['lpips']:.4f}"
            + ("" if m == "identity" else f" ({mean[m]['psnr'] - base['psnr']:+.2f} dB, "
                                          f"{mean[m]['ssim'] - base['ssim']:+.4f}, {mean[m]['lpips'] - base['lpips']:+.4f})")
            for m in ALL))
        print(f"{'':16} PSNR on pixels some training view saw: " + " | ".join(
            f"{m} {mean[m]['psnr_seen']:.2f}" for m in ALL))
    per_view = per_view if per_view in variants else variants[0]
    print(f"\nPer test view ({per_view}): frame number, share of pixels some training view saw, PSNR identity -> "
          f"neighbors (oracle), PSNR on the never-seen pixels, training frames the correction comes from")
    for scene in scenes:
        r = _load(os.path.join(out, scene, "eval", per_view, "heldout_appearance.json"))
        if r is None:
            continue
        for name, pv in sorted(r["test"]["per_view"].items()):
            src = ", ".join(f"{_short(s)} x{w:g}" for s, w in pv["neighbors"].get("from", []))
            uns = pv["identity"].get("psnr_unseen")
            print(f"{scene:9} {_short(name):>8}  frame {pv['frame']:2d}  seen "
                  f"{100 * pv['seen_fraction']:5.1f}%  PSNR {pv['identity']['psnr']:.2f} -> {pv['neighbors']['psnr']:.2f}"
                  f" ({pv['oracle']['psnr']:.2f})  never-seen {'-' if uns is None else f'{uns:.2f}'}  from {src}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="outputs root (contains one folder per scene)")
    p.add_argument("--scenes", required=True, help="comma list")
    p.add_argument("--variants", required=True, help="comma list of trained variants")
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", default=None, help="rasteriser (default: the one the run was trained with)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--report", action="store_true", help="only print the table from existing json files")
    p.add_argument("--per-view", default=None, help="variant whose test views are listed (default: the first)")
    a = p.parse_args()
    scenes, variants = a.scenes.split(","), a.variants.split(",")
    if not a.report:
        from uwsplat.eval.heldout import check_heldout
        for scene in scenes:
            for v in variants:
                d = os.path.join(a.out, scene)
                if not os.path.exists(os.path.join(d, "gs", v, "ckpt.pt")):
                    print(f"skip {scene}/{v}: no checkpoint")
                    continue
                if a.force or not os.path.exists(os.path.join(d, "eval", v, "heldout_appearance.json")):
                    check_heldout(d, v, a.device, a.backend)
    report(a.out, scenes, variants, a.per_view)


if __name__ == "__main__":
    main()
