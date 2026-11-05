"""Interpolated frames between consecutive TRAINING frames, as extra geometry views (no test leakage).

python scripts/make_interframes.py --src Submerged3D --out Submerged3D_ifi [--method raft|farneback]

For every scene: the real frames are copied unchanged and, for each pair of consecutive frames that are both
training frames under the every-8th split (sorted names, test = index % 8 == 0), one midpoint frame is synthesised
from optical flow and saved as <stem of the first>__ifi.png. Pairs that have a test frame between or at either end
are skipped, so no synthetic frame sits at or next to a test pose.

In our pipeline these frames get the split 'aux' (sfm/export.py: AUX_TAG): the SfM registers them (more tracks),
the MVS stage may use them as extra source views when it verifies the training views' depth, and nothing else
does: they are never trained on, never scored, and a point still needs two real training observations.
"""
import argparse
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uwsplat.common.io import list_scene_images, load_rgb, save_rgb  # noqa: E402
from uwsplat.sfm.export import AUX_TAG  # noqa: E402


def flow_farneback(a, b):
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (a, b))
    return cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 5, 21, 5, 7, 1.5, 0)


class RaftFlow:
    def __init__(self, device="cuda"):
        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        self.torch, self.dev = torch, device
        self.w = Raft_Large_Weights.DEFAULT
        self.net = raft_large(weights=self.w).to(device).eval()

    def __call__(self, a, b):
        t = self.torch
        H, W = a.shape[:2]
        ph, pw = (-H) % 8, (-W) % 8
        to = lambda x: t.from_numpy(np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="edge")).permute(2, 0, 1)[None].float() / 255
        x, y = self.w.transforms()(to(a), to(b))
        with t.no_grad():
            f = self.net(x.to(self.dev), y.to(self.dev))[-1][0].permute(1, 2, 0).cpu().numpy()
        return f[:H, :W]


def midpoint(a, b, flow):
    """Frame halfway between a and b: a sampled half a step back along a->b flow, b half a step along b->a."""
    H, W = a.shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    f01, f10 = flow(a, b), flow(b, a)
    wa = cv2.remap(a, gx - 0.5 * f01[..., 0], gy - 0.5 * f01[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wb = cv2.remap(b, gx - 0.5 * f10[..., 0], gy - 0.5 * f10[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return ((wa.astype(np.float32) + wb.astype(np.float32)) / 2 + 0.5).astype(np.uint8)


def train_pairs(names, every=8):
    """Consecutive pairs (k, k+1) of the sorted real frames where both are training frames."""
    return [(k, k + 1) for k in range(len(names) - 1) if k % every != 0 and (k + 1) % every != 0]


def make_scene(src_scene, dst_scene, flow, every=8):
    paths = list_scene_images(src_scene)
    names = [os.path.basename(p) for p in paths]
    out = os.path.join(dst_scene, "input")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for p in paths:
        shutil.copy2(p, out)
    pairs = train_pairs(names, every)
    for k, l in pairs:
        mid = midpoint(load_rgb(paths[k]), load_rgb(paths[l]), flow)
        save_rgb(os.path.join(out, os.path.splitext(names[k])[0] + AUX_TAG + ".png"), mid)
    return len(names), len(pairs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scenes", default="")
    p.add_argument("--method", default="raft", choices=("raft", "farneback"))
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    scenes = a.scenes.split(",") if a.scenes else sorted(d for d in os.listdir(a.src) if os.path.isdir(os.path.join(a.src, d)))
    flow = RaftFlow(a.device) if a.method == "raft" else flow_farneback
    for s in scenes:
        n, m = make_scene(os.path.join(a.src, s), os.path.join(a.out, s), flow)
        print(f"{s}: {n} real frames + {m} interpolated ({a.method})", flush=True)


if __name__ == "__main__":
    main()
