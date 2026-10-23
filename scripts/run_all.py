"""Run the full pipeline on every Submerged3D scene, then summarise.

python scripts/run_all.py --data Submerged3D --out outputs
python scripts/run_all.py --data Submerged3D --out outputs --variants full,sparse_init,dense_init,no_medium,baseline_3dgs
Stages whose outputs already exist are skipped unless --force is given.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uwsplat.common.io import log  # noqa: E402
from uwsplat.pipeline import VARIANTS, parse_sets, run_scene  # noqa: E402

import summarize  # noqa: E402

DONE = {"sfm": "sfm/scene.json", "dense": "dense/summary.json", "fusion": "fusion/fused.ply"}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="Submerged3D")
    p.add_argument("--out", default="outputs")
    p.add_argument("--scenes", default="", help="comma list (default: every folder in --data)")
    p.add_argument("--variants", default="full", help=f"comma list from {sorted(VARIANTS)}")
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    a = p.parse_args()
    scenes = a.scenes.split(",") if a.scenes else sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d)))
    variants = a.variants.split(",")
    for scene in scenes:
        out = os.path.join(a.out, scene)
        stages = [s for s in ("sfm", "dense", "fusion") if a.force or not os.path.exists(os.path.join(out, DONE[s]))]
        todo = [v for v in variants if a.force or not os.path.exists(os.path.join(out, "eval", v, "metrics.json"))]
        if todo:
            stages += ["train", "eval"]
        log(f"##### {scene}: stages {stages}, variants {todo}")
        if stages:
            run_scene(os.path.join(a.data, scene), out, stages, parse_sets(a.set), todo or variants, a.device)
    summarize.main(a.out)
