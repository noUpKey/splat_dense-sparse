"""Run (a subset of) the pipeline on one scene.

python scripts/run_scene.py --scene Submerged3D/Tokai --out outputs/Tokai
python scripts/run_scene.py --scene Submerged3D/Tokai --out outputs/Tokai --stages train,eval \
       --variants full,sparse_init --set train.iterations=20000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uwsplat.pipeline import STAGES, VARIANTS, parse_sets, run_scene  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True, help="scene folder containing input/ (e.g. Submerged3D/Tokai)")
    p.add_argument("--out", required=True, help="output folder for this scene")
    p.add_argument("--stages", default=",".join(STAGES), help=f"comma list from {STAGES}")
    p.add_argument("--variants", default="full", help=f"comma list from {sorted(VARIANTS)}")
    p.add_argument("--device", default="cuda")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="config override, sections: sfm, dense, fusion, mvs, train (repeatable)")
    a = p.parse_args()
    run_scene(a.scene, a.out, a.stages.split(","), parse_sets(a.set), a.variants.split(","), a.device)
