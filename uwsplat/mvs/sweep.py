"""Multi-view photometric depth verification with known SfM poses (prior-guided plane sweep).

For each pixel p of a reference training view, depth hypotheses are placed around a centre depth
d0(p) at log offsets half(p) * tau_k, tau_k evenly spaced in [-1, 1]:
  * prior pass: d0 = the dense network's SfM-scaled depth, half = +-0.2 (~ +-22 %);
  * wide pass:  d0 / half span the view's SIFT depth range, for pixels the prior pass missed.
Every source view is warped into the reference view at every hypothesis with the known poses, and the
windowed zero-mean normalised cross-correlation (NCC) with the reference is computed. Each window
pixel uses its own hypothesis depth, so the window follows the prior surface rather than a
fronto-parallel plane.

Why NCC suits water: between two views at similar range, the medium changes a surface patch by a
gain e^(-beta_D r) and an offset B_inf (1 - e^(-beta_B r)); NCC is invariant to exactly that. What the
water really removes is texture contrast (signal below the noise), which the texture gate catches:
a reference window whose standard deviation is below `min_std` grey levels is never verified. That
rejects open water (noise-flat) without any learned confidence.
"""
import torch
import torch.nn.functional as F


def box(x: torch.Tensor, k: int) -> torch.Tensor:
    """k x k mean over the in-image part of the window. x (B, C, h, w)."""
    return F.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)


def pixel_rays(K: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(3, h*w) camera rays with z = 1 for pixel centres (+0.5 convention)."""
    v, u = torch.meshgrid(torch.arange(h, device=K.device, dtype=K.dtype) + 0.5,
                          torch.arange(w, device=K.device, dtype=K.dtype) + 0.5, indexing="ij")
    return torch.linalg.solve(K, torch.stack([u, v, torch.ones_like(u)]).reshape(3, -1))


def window_std(gray: torch.Tensor, k: int) -> torch.Tensor:
    """Standard deviation of the k x k window around each pixel. gray (h, w) -> (h, w)."""
    g = gray[None, None]
    return (box(g * g, k) - box(g, k) ** 2).clamp_min(0).sqrt()[0, 0]


def masked_ncc(ref: torch.Tensor, warped: torch.Tensor, valid: torch.Tensor, k: int, min_valid: float = 0.9,
               eps: float = 1.0) -> torch.Tensor:
    """Windowed NCC over the samples that landed inside the source image.

    ref (1, 1, h, w), warped / valid (B, 1, h, w). Returns (B, 1, h, w) in [-1, 1]; -1 where the
    centre sample is invalid or fewer than `min_valid` of the window samples are. eps (grey^4)
    keeps flat windows near 0 instead of dividing by ~0."""
    n = box(valid, k)
    nz = n.clamp_min(1e-6)
    rv, wv = ref * valid, warped * valid
    mr, mw = box(rv, k) / nz, box(wv, k) / nz
    vr = (box(rv * ref, k) / nz - mr ** 2).clamp_min(0)
    vw = (box(wv * warped, k) / nz - mw ** 2).clamp_min(0)
    cov = box(rv * warped, k) / nz - mr * mw
    ncc = cov / torch.sqrt(vr * vw + eps)
    return torch.where((n >= min_valid) & (valid > 0), ncc, torch.full_like(ncc, -1.0))


@torch.no_grad()
def ncc_volume(ref: torch.Tensor, srcs: torch.Tensor, K: torch.Tensor, rel: torch.Tensor, d0: torch.Tensor,
               half: torch.Tensor, n_hyp: int, window: int = 7, min_valid: float = 0.9, chunk: int = 16):
    """NCC of every source view at every depth hypothesis.

    ref (h, w) and srcs (S, h, w): grey images on the same grid; K (3, 3) intrinsics of that grid;
    rel (S, 3, 4): source-from-reference rigid transforms [R | t]; d0, half (h, w): centre depth (> 0)
    and log half-range of the hypotheses. Returns ncc (S, D, h, w) and tau (D,)."""
    S, (h, w) = srcs.shape[0], ref.shape
    tau = torch.linspace(-1.0, 1.0, n_hyp, device=ref.device, dtype=ref.dtype)
    rays = pixel_rays(K, h, w)
    r = ref[None, None]
    d0f, hf = d0.reshape(1, -1), half.reshape(1, -1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    out = torch.full((S, n_hyp, h, w), -1.0, device=ref.device, dtype=ref.dtype)
    for s in range(S):
        a = rel[s, :, :3] @ rays                       # (3, hw): rotated rays
        t = rel[s, :, 3]
        src = srcs[s][None, None]
        for k0 in range(0, n_hyp, chunk):
            tk = tau[k0:k0 + chunk]
            B = len(tk)
            d = d0f * torch.exp(hf * tk[:, None])      # (B, hw) hypothesis depths
            X = d[:, None] * a[None] + t[None, :, None]  # (B, 3, hw) in the source camera
            z = X[:, 2]
            zc = z.clamp_min(1e-6)
            u = fx * X[:, 0] / zc + cx
            v = fy * X[:, 1] / zc + cy
            ok = (z > 1e-6) & (u >= 0) & (u <= w) & (v >= 0) & (v <= h)
            grid = torch.stack([u / w * 2 - 1, v / h * 2 - 1], -1).reshape(B, h, w, 2)
            warped = F.grid_sample(src.expand(B, -1, -1, -1), grid, mode="bilinear", padding_mode="zeros",
                                   align_corners=False)
            out[s, k0:k0 + B] = masked_ncc(r, warped, ok.reshape(B, 1, h, w).to(ref.dtype), window, min_valid)[:, 0]
    return out, tau


def best_depth(ncc: torch.Tensor, tau: torch.Tensor, d0: torch.Tensor, half: torch.Tensor, topk: int = 2,
               min_ncc: float = 0.5):
    """Aggregate the sources (mean of the k best NCCs per hypothesis, robust to one occluded or
    out-of-view source), take the best hypothesis and refine it with a parabola.

    Returns depth, score (aggregated NCC at the peak) and valid (score >= min_ncc, interior,
    concave peak), all (h, w)."""
    S, D = ncc.shape[:2]
    k = min(topk, S)
    agg = ncc.mean(0) if k == S else ncc.topk(k, dim=0).values.mean(0)  # (D, h, w)
    score, idx = agg.max(0)
    c0 = agg.gather(0, (idx - 1).clamp(0, D - 1)[None])[0]
    c2 = agg.gather(0, (idx + 1).clamp(0, D - 1)[None])[0]
    den = c0 - 2 * score + c2
    concave = den < -1e-6
    off = torch.where(concave, 0.5 * (c0 - c2) / torch.where(concave, den, -torch.ones_like(den)),
                      torch.zeros_like(den)).clamp(-0.5, 0.5)
    t = tau[idx] + off * (tau[1] - tau[0])
    depth = d0 * torch.exp(half * t)
    valid = (score >= min_ncc) & (idx > 0) & (idx < D - 1) & concave
    return depth, score, valid
