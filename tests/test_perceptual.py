"""Perceptual (LPIPS) training loss: frozen network, gradients only into the render, seeded random crops.

python -m tests.test_perceptual
Needs the lpips package. The backbone is built without its pretrained weights here (pnet_rand), so the test
downloads nothing; the trainer uses the pretrained network through exactly the same code path.
"""
import lpips
import torch

from uwsplat.gs.losses import PerceptualLoss

_LPIPS = lpips.LPIPS


def _random_backbone(*args, **kwargs):
    return _LPIPS(*args, **{**kwargs, "pnet_rand": True})


def main():
    lpips.LPIPS = _random_backbone
    try:
        torch.manual_seed(0)
        H, W, c = 120, 200, 64
        gt = torch.rand(H, W, 3)
        f = PerceptualLoss("vgg", crop=c, device="cpu", seed=0)
        assert not any(p.requires_grad for p in f.fn.parameters()), "LPIPS network must be frozen"
        assert float(f(gt, gt)) == 0.0
        pred = (gt + 0.1 * torch.randn_like(gt)).clamp(0, 1).requires_grad_(True)
        loss = f(pred, gt)
        loss.backward()
        assert float(loss) > 0 and pred.grad is not None
        ys, xs = torch.nonzero(pred.grad.abs().sum(-1) > 0, as_tuple=True)
        assert int(ys.max() - ys.min()) < c and int(xs.max() - xs.min()) < c, "gradient must stay inside the crop"
        first = [PerceptualLoss("vgg", c, "cpu", seed=3).window(H, W) for _ in range(2)]
        assert first[0] == first[1], "crops must be reproducible for a given seed"
        g = PerceptualLoss("vgg", c, "cpu", seed=3)
        wins = [g.window(H, W) for _ in range(50)]
        assert all(0 <= y and y + h <= H and 0 <= x and x + w <= W and (h, w) == (c, c) for y, x, h, w in wins)
        assert len(set(wins)) > 10, "crops should move around the image"
        assert PerceptualLoss("vgg", 0, "cpu").window(H, W) == (0, 0, H, W)    # crop 0 = whole image
        assert PerceptualLoss("vgg", 512, "cpu").window(H, W) == (0, 0, H, W)  # crop larger than the image
        print(f"perceptual loss test passed (LPIPS of a noisy copy {float(loss):.4f}, {len(set(wins))} distinct crops)")
    finally:
        lpips.LPIPS = _LPIPS


if __name__ == "__main__":
    main()
