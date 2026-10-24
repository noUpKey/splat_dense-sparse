"""Losses of the medium-aware splatting stage.

Images are (H, W, 3) tensors, depth/alpha (H, W, 1).
  reconstruction : (1 - l) L1 + l (1 - SSIM) on the composited underwater image
  depth          : edge-aware smoothness of rendered depth (fixes SeaSplat's signed image
                   gradient), dense prior on the water region, SIFT keypoint depths on the wreck
  RUSplatting    : edge-aware smoothness of the rendered *image*, switched off across rendered
                   depth edges (RUSplatting's EdgeSmoothnessLoss, "ESL")
  perceptual     : LPIPS between the rendered training image and the photo (frozen VGG / AlexNet),
                   switched on late in training by the trainer
  colour         : grey-world and saturation priors on the restored radiance J
  medium         : dark-pixel backscatter prior, background opacity prior
"""
import torch
import torch.nn.functional as F

_WINDOWS = {}


def _window(device, size=11, sigma=1.5):
    key = (device, size)
    if key not in _WINDOWS:
        g = torch.exp(-((torch.arange(size, device=device) - size // 2) ** 2).float() / (2 * sigma ** 2))
        g = g / g.sum()
        _WINDOWS[key] = (g[:, None] @ g[None]).expand(3, 1, size, size).contiguous()
    return _WINDOWS[key]


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """3DGS SSIM (11x11 Gaussian window, sigma 1.5) of two (H, W, 3) images."""
    x, y = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]
    w = _window(x.device)
    mu_x, mu_y = F.conv2d(x, w, padding=5, groups=3), F.conv2d(y, w, padding=5, groups=3)
    sxx = F.conv2d(x * x, w, padding=5, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=5, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=5, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return m.mean()


def reconstruction(pred, gt, lambda_dssim=0.2):
    return (1 - lambda_dssim) * (pred - gt).abs().mean() + lambda_dssim * (1 - ssim(pred, gt))


def edge_aware_smoothness(depth, alpha, img, gamma=10.0):
    """|grad(disparity / mean disparity)| * exp(-gamma |grad I|), on pixels covered by Gaussians."""
    m = (alpha > 0.5).float()
    disp = m / depth.clamp_min(1e-6)
    disp = disp / (disp.sum() / m.sum().clamp_min(1)).clamp_min(1e-8)
    gx = (disp[:, 1:] - disp[:, :-1]).abs() * (m[:, 1:] * m[:, :-1])
    gy = (disp[1:] - disp[:-1]).abs() * (m[1:] * m[:-1])
    wx = torch.exp(-gamma * (img[:, 1:] - img[:, :-1]).abs().mean(-1, keepdim=True))
    wy = torch.exp(-gamma * (img[1:] - img[:-1]).abs().mean(-1, keepdim=True))
    return (gx * wx).mean() + (gy * wy).mean()


def esl_smoothness(image, depth, alpha, beta=5.0, detach=True):
    """RUSplatting's edge-aware smoothness loss (utils/loss_utils.py EdgeSmoothnessLoss, called with
    beta=5 and weight 0.05 in train.py; the paper writes lambda_s = 0.2):

        mean |d_x I| exp(-beta |d_x D|) + mean |d_y I| exp(-beta |d_y D|)

    I is the rendered underwater image, D the rendered depth. Their rasteriser outputs accumulated
    depth (sum z alpha T, 0 where nothing is drawn), so coverage boundaries count as depth edges; here
    D = expected depth x alpha, divided by the mean depth of the covered pixels, which makes beta
    independent of the scene scale (uwsplat scenes have median range 1, so the division is ~1).

    detach=True (default) keeps the weights out of the gradient: in the original, lowering the loss by
    *creating* depth jumps at texture edges is also rewarded. detach=False reproduces RUSplatting."""
    m = (alpha > 0.5).float()
    mean_depth = ((depth * m).sum() / m.sum().clamp_min(1.0)).detach().clamp_min(1e-6)
    D = depth * alpha / mean_depth
    if detach:
        D = D.detach()
    wx = torch.exp(-beta * (D[:, 1:] - D[:, :-1]).abs())
    wy = torch.exp(-beta * (D[1:] - D[:-1]).abs())
    gx = (image[:, 1:] - image[:, :-1]).abs()
    gy = (image[1:] - image[:-1]).abs()
    return (gx * wx).mean() + (gy * wy).mean()


class PerceptualLoss:
    """LPIPS (lpips package, v0.1 linear heads) between a rendered training image and its photo.

    The network is frozen (no parameter receives gradients); only the render does. Each call scores one
    random crop x crop window, the same for both images (crop 0 = whole image), drawn from a seeded
    generator so runs are reproducible. net="vgg" is the network the evaluation reports; net="alex" gives
    a perceptual loss that is not the evaluation network itself."""

    def __init__(self, net: str = "vgg", crop: int = 512, device="cuda", seed: int = 0):
        import lpips
        self.fn = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        for p in self.fn.parameters():
            p.requires_grad_(False)
        self.crop = crop
        self.gen = torch.Generator().manual_seed(seed)

    def window(self, H: int, W: int):
        if not self.crop or (H <= self.crop and W <= self.crop):
            return 0, 0, H, W
        h, w = min(self.crop, H), min(self.crop, W)
        y = int(torch.randint(0, H - h + 1, (1,), generator=self.gen))
        x = int(torch.randint(0, W - w + 1, (1,), generator=self.gen))
        return y, x, h, w

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """pred, gt (H, W, 3) in [0, 1] -> scalar LPIPS of the crop."""
        y, x, h, w = self.window(*pred.shape[:2])
        to = lambda im: im[y:y + h, x:x + w].permute(2, 0, 1)[None] * 2 - 1
        return self.fn(to(pred), to(gt)).mean()


# Masked means below are written without Python branches on tensor values, which would
# force a GPU synchronisation every iteration (and an empty selection would give NaN).

def dense_depth_prior(depth, alpha, target, weight):
    """Confidence-weighted |log depth - log prior| over the water-surface pixels of the dense branch."""
    w = weight[..., None] * (alpha.detach() > 0.1) * (target[..., None] > 0)
    err = (torch.log(depth.clamp_min(1e-6)) - torch.log(target[..., None].clamp_min(1e-6))).abs()
    return (w * err).sum() / w.sum().clamp_min(1e-6)


def sparse_depth_prior(depth, alpha, uv, z, delta=0.05):
    """Huber on log depth at SIFT keypoints (wreck). uv in +0.5 pixel convention of this image."""
    H, W = depth.shape[:2]
    g = torch.stack([uv[:, 0] / W * 2 - 1, uv[:, 1] / H * 2 - 1], -1)[None, None]
    both = torch.cat([depth, alpha], -1).permute(2, 0, 1)[None]
    d, a = F.grid_sample(both, g, mode="bilinear", align_corners=False)[0, :, 0]
    ok = ((a.detach() > 0.1) & (d > 0)).float()
    e = F.smooth_l1_loss(torch.log(d.clamp_min(1e-6)), torch.log(z), beta=delta, reduction="none")
    return (ok * e).sum() / ok.sum().clamp_min(1.0)


def gray_world(J, alpha, target=0.5):
    m = (alpha > 0.5).float()
    mean = (J * m).sum((0, 1)) / (alpha * m).sum().clamp_min(1)
    return ((mean - target) ** 2).mean() * (m.sum() > 0)


def saturation(J, vmax=1.0):
    return (F.relu(J - vmax) ** 2).mean()


def backscatter_prior(gt, depth, alpha, ray_norm, binf_map, beta_b, kappa=10.0):
    """DeepSeeColor/SeaSplat dark-pixel prior: backscatter should explain the darkest content but
    never exceed the image. Geometry is detached, so only beta_B and B_inf receive gradients."""
    r = (depth * ray_norm).detach()
    a = alpha.detach()
    B = a * binf_map * (1 - torch.exp(-beta_b * r)) + (1 - a) * binf_map
    direct = gt - B
    return F.relu(direct).mean() + kappa * F.smooth_l1_loss(F.relu(-direct), torch.zeros_like(direct), beta=0.2)


def background_opacity(alpha, gt, binf_map, thresh=0.08):
    """Pixels the medium alone explains (|gt - B_inf(d)| small) should be left transparent."""
    m = ((gt - binf_map.detach()).norm(dim=-1, keepdim=True) < thresh).float()
    return (alpha * m).sum() / m.sum().clamp_min(1.0)
