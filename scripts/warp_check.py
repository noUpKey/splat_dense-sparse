"""Training-free check of the water-aware image-based residual idea. No training.

python scripts/warp_check.py --out outputs_v2 --scenes Tokai,Cormoran,Isro,Kwaj --tag mvs_lpips
python scripts/warp_check.py --out outputs_seathru --scenes ... --sources overlap --target_colour identity
python scripts/warp_check.py --out outputs_v2 --scenes ... --report          # print the table and the decision only

Warps the training photos next to each test view into it with the model's depth and scores six variants on the pixels
every source sees: naive photo copy, camera transfer, camera + water transfer, the render, and render + transferred
residual (water-scaled / unscaled). Writes <out>/<scene>/eval/<tag>/warp_check.json and montages in
eval/<tag>/warp_check/ (photo | render | camera+water warp | render+residual | mask). The pre-registered rule is in
uwsplat/eval/warp_check.py. Video transects (Submerged3D): --sources temporal --target_colour neighbors (default).
Photo collections (SeaThru-NeRF): --sources overlap --target_colour identity.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uwsplat.eval.warp_check import VARIANTS, decide  # noqa: E402

LETTER = dict(zip(VARIANTS, "abcdef"))


def report(out, scenes, tag):
    res = {}
    for sc in scenes:
        p = os.path.join(out, sc, "eval", tag, "warp_check.json")
        if os.path.exists(p):
            res[sc] = json.load(open(p))
    if not res:
        print("no results")
        return {}
    print("Masked PSNR (pixels every source sees) and, in brackets, high-frequency correlation")
    print(f"{'scene':10}" + "".join(f"{LETTER[v] + ' ' + v:>24}" for v in VARIANTS))
    for sc, r in res.items():
        s = r["summary"]
        if not s:
            print(f"{sc:10} (no scored views)")
            continue
        print(f"{sc:10}" + "".join(f"{s['masked'][v]['psnr']:15.2f} ({s['masked'][v]['hf_corr']:.3f})"
                                    for v in VARIANTS))
    print("\nFull image, composite (variant inside the mask, render outside): PSNR / SSIM / LPIPS")
    for sc, r in res.items():
        s = r["summary"]
        if s:
            print(f"{sc:10}" + " | ".join(f"{LETTER[v]} {s['full_composite'][v]['psnr']:.2f} / "
                                          f"{s['full_composite'][v]['ssim']:.4f} / {s['full_composite'][v]['lpips']:.4f}"
                                          for v in VARIANTS))
    print("\nWater term and coverage")
    for sc, r in res.items():
        s = r["summary"]
        if s:
            print(f"{sc:10} mask {100 * s['mask_fraction']:5.1f} %   range change median "
                  f"{100 * s['rel_range_change_median']:5.1f} %   (c)-(b) {s['water_minus_camera_8bit_mean']:5.2f} "
                  f"8-bit levels   physics (c-b) {s['physics_delta_db']:+.2f} dB   detail (c-d) "
                  f"{s['detail_delta_db']:+.2f} dB / {s['detail_delta_hf']:+.3f} HF")
    d = decide({sc: r["summary"] for sc, r in res.items()})
    print(f"\nPre-registered rule over {d['n_scenes']} scene(s), need {d['needed']}:"
          f"\n  physics  (c) > (b) by > 0.1 dB : {d['physics_pass']} -> {'PASS' if d['physics'] else 'FAIL'}"
          f"\n  detail   (c) > (d) masked PSNR : {d['detail_psnr_pass']} -> {'PASS' if d['detail_psnr'] else 'FAIL'}"
          f"\n  detail   (c) > (d) HF corr     : {d['detail_hf_pass']} -> {'PASS' if d['detail_hf'] else 'FAIL'}"
          f" (secondary)"
          f"\n  build the residual network     : {'YES' if d['build_network'] else 'NO'}")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="outputs root (one folder per scene)")
    ap.add_argument("--scenes", required=True, help="comma list")
    ap.add_argument("--tag", default="mvs_lpips")
    ap.add_argument("--sources", default="temporal", choices=("temporal", "overlap"))
    ap.add_argument("--k", type=int, default=2, help="number of sources for --sources overlap")
    ap.add_argument("--target_colour", default="neighbors", choices=("neighbors", "identity"))
    ap.add_argument("--alpha_min", type=float, default=0.95)
    ap.add_argument("--occ_tol", type=float, default=0.03, help="relative depth tolerance of the occlusion test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--no_images", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    scenes = a.scenes.split(",")
    if not a.report:
        from uwsplat.eval.warp_check import run
        for sc in scenes:
            d = os.path.join(a.out, sc)
            if not os.path.exists(os.path.join(d, "gs", a.tag, "ckpt.pt")):
                print(f"skip {sc}: no checkpoint")
                continue
            if a.force or not os.path.exists(os.path.join(d, "eval", a.tag, "warp_check.json")):
                run(d, a.tag, a.device, a.backend, a.sources, a.k, a.target_colour, a.alpha_min, a.occ_tol,
                    save_images=not a.no_images)
    report(a.out, scenes, a.tag)


if __name__ == "__main__":
    main()
