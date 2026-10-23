"""Collect eval/<variant>/metrics.json of all scenes into a results table (markdown + json).

python scripts/summarize.py --out outputs
"""
import argparse
import json
import os

import numpy as np

# Published / previously measured Submerged3D averages, for context (not re-run here).
REFERENCE = {
    "RUSplatting (BMVC'25, paper)": (25.80, 0.7724, 0.3008),
    "COLMAP -> 3DGS (Caldeira et al. '26, 720p)": (23.48, 0.7385, 0.3333),
    "COLMAP -> RUSplatting (Caldeira et al. '26)": (22.97, 0.7114, 0.3514),
    "Geometry-guided (Caldeira et al. '26)": (23.49, 0.7045, 0.3322),
}


def collect(out_root):
    rows = {}
    for scene in sorted(os.listdir(out_root)):
        ev = os.path.join(out_root, scene, "eval")
        if not os.path.isdir(ev):
            continue
        for tag in sorted(os.listdir(ev)):
            f = os.path.join(ev, tag, "metrics.json")
            if os.path.exists(f):
                with open(f) as fh:
                    rows.setdefault(tag, {})[scene] = json.load(fh)
    return rows


def main(out_root):
    rows = collect(out_root)
    scenes = sorted({s for v in rows.values() for s in v})
    lines = ["| Variant | " + " | ".join(scenes) + " | **Average** PSNR / SSIM / LPIPS |",
             "|---|" + "---|" * (len(scenes) + 1)]
    table = {}
    for tag, per in rows.items():
        cells = []
        for s in scenes:
            m = per.get(s, {}).get("mean")
            cells.append(f"{m['psnr']:.2f} / {m['ssim']:.3f} / {m['lpips']:.3f}" if m else "-")
        done = [per[s]["mean"] for s in scenes if s in per]
        avg = {k: float(np.mean([m[k] for m in done])) for k in ("psnr", "ssim", "lpips")} if done else None
        table[tag] = {"per_scene": {s: per[s]["mean"] for s in per}, "average": avg, "n_scenes": len(done)}
        avg_s = f"**{avg['psnr']:.2f} / {avg['ssim']:.4f} / {avg['lpips']:.4f}**" if avg else "-"
        lines.append(f"| {tag} | " + " | ".join(cells) + f" | {avg_s} |")
    lines += ["", "Reference averages on Submerged3D (reported, not re-run):", ""]
    lines += [f"- {k}: PSNR {p:.2f} / SSIM {s:.4f} / LPIPS {l:.4f}" for k, (p, s, l) in REFERENCE.items()]
    region = []
    for tag, per in rows.items():
        vals = [per[s]["mean"] for s in per if "psnr_wreck" in per[s]["mean"]]
        if vals:
            region.append(f"- {tag}: wreck PSNR {np.mean([v['psnr_wreck'] for v in vals]):.2f}, "
                          f"water PSNR {np.mean([v['psnr_water'] for v in vals]):.2f}")
    if region:
        lines += ["", "Region-wise PSNR (test views, wreck = SIFT-supported pixels):", ""] + region
    md = "\n".join(lines)
    print(md)
    with open(os.path.join(out_root, "results.md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(out_root, "results.json"), "w") as f:
        json.dump(table, f, indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs")
    main(p.parse_args().out)
