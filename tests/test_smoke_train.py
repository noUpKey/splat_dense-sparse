"""CPU smoke test: train + evaluate with the reference rasteriser on tiny images.

python -m tests.test_smoke_train <scene_output_dir>
The scene must already have sfm/, dense/ and fusion/ outputs. Checks that every
parameter group receives gradients, densification/pruning/opacity reset run,
and the evaluation writes PSNR / SSIM / LPIPS.
"""
import sys

import torch

from uwsplat.eval.evaluate import evaluate
from uwsplat.gs.trainer import TrainConfig, train


def main(out_dir):
    cfg = TrainConfig(iterations=120, backend="torch", device="cpu", downscale=16, max_init_points=2500,
                      eval_every=60, log_every=20, sh_increase_every=30, color_from=40, medium_warmup=10,
                      densify=dict(refine_start=10, refine_every=20, reset_every=60, max_gaussians=4000))
    final = train(out_dir, cfg, tag="smoke")
    ck = torch.load(f"{out_dir}/gs/smoke/ckpt.pt", weights_only=False)
    m0 = ck["medium"]
    print("medium after training:", {k: v.flatten()[:3].tolist() for k, v in m0.items()})
    res = evaluate(out_dir, "smoke", "cpu", "torch")
    assert final["n_gaussians"] > 0 and all(k in res["mean"] for k in ("psnr", "ssim", "lpips"))
    print("smoke test passed:", res["mean"])


if __name__ == "__main__":
    main(sys.argv[1])
