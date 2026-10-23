"""Dense branch network: a two-view pointmap transformer, re-implemented and pruned.

Reference design (task figure): shared ViT-L encoder (16x16 patches, 24 blocks,
16 heads, 2D RoPE) -> 1024->768 projection -> two 12-block decoders that
cross-attend to each other's previous-block tokens -> one DPT head per view,
reading token sets {0, 6, 9, 12} -> pointmaps (v/|v| * (e^|v| - 1)) and
confidences (1 + e^c), both expressed in camera-1 coordinates.

Differences, because camera poses already come from the SIFT SfM branch:
  * the encoder runs once per image and its tokens are cached and reused by
    every pair (the reference re-encodes both images for every pair);
  * only the reference-view head is built: the second head predicts the other
    image's points in this camera's frame, which known poses make redundant
    (20.2M parameters never loaded);
  * the source decoder's last block feeds nothing but that second head, so it
    is dropped too (9.5M parameters);
  * attention uses fused scaled-dot-product kernels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE2D(nn.Module):
    """2D rotary position embedding: the first half of each head's channels is rotated
    by the token's row index, the second half by its column index (base frequency 100)."""

    def __init__(self, base: float = 100.0):
        super().__init__()
        self.base = base
        self._cache = {}

    def _cos_sin(self, d, n, device, dtype):
        key = (d, n, device, dtype)
        if key not in self._cache:
            inv = 1.0 / (self.base ** (torch.arange(0, d, 2, device=device).float() / d))
            ang = torch.arange(n, device=device).float()[:, None] * inv[None]
            ang = torch.cat([ang, ang], -1).to(dtype)
            self._cache[key] = (ang.cos(), ang.sin())
        return self._cache[key]

    @staticmethod
    def _rotate_half(x):
        a, b = x.chunk(2, -1)
        return torch.cat([-b, a], -1)

    def forward(self, x, pos):
        """x: (B, heads, N, D); pos: (B, N, 2) integer (row, col)."""
        cos, sin = self._cos_sin(x.shape[-1] // 2, int(pos.max()) + 1, x.device, x.dtype)
        parts = []
        for axis, part in enumerate(x.chunk(2, -1)):
            c, s = cos[pos[..., axis]][:, None], sin[pos[..., axis]][:, None]
            parts.append(part * c + self._rotate_half(part) * s)
        return torch.cat(parts, -1)


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, rope):
        super().__init__()
        self.heads, self.rope = heads, rope
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)

    def forward(self, x, pos):
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(self.rope(q, pos), self.rope(k, pos), v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, rope):
        super().__init__()
        self.heads, self.rope = heads, rope
        self.projq, self.projk, self.projv = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, xq, xkv, qpos, kpos):
        B, Nq, C = xq.shape
        h = self.heads
        q = self.projq(xq).view(B, Nq, h, C // h).transpose(1, 2)
        k = self.projk(xkv).view(B, -1, h, C // h).transpose(1, 2)
        v = self.projv(xkv).view(B, -1, h, C // h).transpose(1, 2)
        x = F.scaled_dot_product_attention(self.rope(q, qpos), self.rope(k, kpos), v)
        return self.proj(x.transpose(1, 2).reshape(B, Nq, C))


class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, rope):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn, self.mlp = SelfAttention(dim, heads, rope), Mlp(dim, 4 * dim)

    def forward(self, x, pos):
        x = x + self.attn(self.norm1(x), pos)
        return x + self.mlp(self.norm2(x))


class DecoderBlock(nn.Module):
    """Self-attention on own tokens, cross-attention to the other view's tokens, MLP."""

    def __init__(self, dim, heads, rope):
        super().__init__()
        self.norm1, self.norm2, self.norm3, self.norm_y = (nn.LayerNorm(dim, eps=1e-6) for _ in range(4))
        self.attn, self.cross_attn = SelfAttention(dim, heads, rope), CrossAttention(dim, heads, rope)
        self.mlp = Mlp(dim, 4 * dim)

    def forward(self, x, y, xpos, ypos):
        x = x + self.attn(self.norm1(x), xpos)
        y = self.norm_y(y)
        x = x + self.cross_attn(self.norm2(x), y, xpos, ypos)
        return x + self.mlp(self.norm3(x))


class ResidualConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(F.relu(x))))


class FusionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.res1, self.res2 = ResidualConv(c), ResidualConv(c)
        self.out_conv = nn.Conv2d(c, c, 1)

    def forward(self, x, lateral=None):
        if lateral is not None:
            x = x + self.res1(lateral)
        x = F.interpolate(self.res2(x), scale_factor=2, mode="bilinear", align_corners=True)
        return self.out_conv(x)


class DPTHead(nn.Module):
    """Reassemble four token sets at 1/4, 1/8, 1/16, 1/32 scale, fuse coarse-to-fine, predict 4 channels."""

    def __init__(self, enc_dim=1024, dec_dim=768, dims=(96, 192, 384, 768), feat=256, out_ch=4):
        super().__init__()
        d_in = (enc_dim, dec_dim, dec_dim, dec_dim)
        self.act_postprocess = nn.ModuleList([
            nn.Sequential(nn.Conv2d(d_in[0], dims[0], 1), nn.ConvTranspose2d(dims[0], dims[0], 4, stride=4)),
            nn.Sequential(nn.Conv2d(d_in[1], dims[1], 1), nn.ConvTranspose2d(dims[1], dims[1], 2, stride=2)),
            nn.Sequential(nn.Conv2d(d_in[2], dims[2], 1)),
            nn.Sequential(nn.Conv2d(d_in[3], dims[3], 1), nn.Conv2d(dims[3], dims[3], 3, stride=2, padding=1)),
        ])
        self.layer_rn = nn.ModuleList([nn.Conv2d(d, feat, 3, padding=1, bias=False) for d in dims])
        self.refinenet = nn.ModuleList([FusionBlock(feat) for _ in range(4)])
        self.out = nn.Sequential(
            nn.Conv2d(feat, feat // 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(feat // 2, 128, 3, padding=1), nn.ReLU(True), nn.Conv2d(128, out_ch, 1))

    def forward(self, taps, h, w):
        x = [t.transpose(1, 2).reshape(t.shape[0], -1, h, w) for t in taps]
        x = [rn(pp(t)) for t, pp, rn in zip(x, self.act_postprocess, self.layer_rn)]
        path = self.refinenet[3](x[3])[:, :, :x[2].shape[2], :x[2].shape[3]]
        path = self.refinenet[2](path, x[2])
        path = self.refinenet[1](path, x[1])
        path = self.refinenet[0](path, x[0])
        return self.out(path)


class PointmapNet(nn.Module):
    def __init__(self, enc_dim=1024, enc_depth=24, enc_heads=16, dec_dim=768, dec_depth=12, dec_heads=12,
                 patch=16, rope_base=100.0):
        super().__init__()
        self.patch = patch
        self.rope = RoPE2D(rope_base)
        self.patch_embed = nn.Conv2d(3, enc_dim, patch, stride=patch)
        self.enc_blocks = nn.ModuleList([EncoderBlock(enc_dim, enc_heads, self.rope) for _ in range(enc_depth)])
        self.enc_norm = nn.LayerNorm(enc_dim, eps=1e-6)
        self.decoder_embed = nn.Linear(enc_dim, dec_dim)
        self.ref_blocks = nn.ModuleList([DecoderBlock(dec_dim, dec_heads, self.rope) for _ in range(dec_depth)])
        self.src_blocks = nn.ModuleList([DecoderBlock(dec_dim, dec_heads, self.rope) for _ in range(dec_depth - 1)])
        self.dec_norm = nn.LayerNorm(dec_dim, eps=1e-6)
        self.head = DPTHead(enc_dim, dec_dim)
        self.hooks = (0, dec_depth // 2, dec_depth * 3 // 4, dec_depth)

    def encode(self, img):
        """img: (B, 3, H, W) in [-1, 1] with H, W multiples of 16 -> tokens (B, N, 1024), positions (B, N, 2)."""
        B, _, H, W = img.shape
        h, w = H // self.patch, W // self.patch
        x = self.patch_embed(img).flatten(2).transpose(1, 2)
        pos = torch.cartesian_prod(torch.arange(h, device=img.device), torch.arange(w, device=img.device))
        pos = pos[None].expand(B, -1, -1)
        for blk in self.enc_blocks:
            x = blk(x, pos)
        return self.enc_norm(x), pos

    def decode(self, f_ref, f_src, pos_ref, pos_src):
        """Token sets of the reference view at the head's hook indices."""
        x, y = self.decoder_embed(f_ref), self.decoder_embed(f_src)
        taps = [f_ref]
        for k, blk in enumerate(self.ref_blocks):
            x_new = blk(x, y, pos_ref, pos_src)
            if k < len(self.src_blocks):   # both decoders read the other's tokens from block k-1
                y = self.src_blocks[k](y, x, pos_src, pos_ref)
            x = x_new
            taps.append(x)
        taps[-1] = self.dec_norm(taps[-1])
        return [taps[i] for i in self.hooks]

    def pointmap(self, f_ref, f_src, pos_ref, pos_src, H, W):
        """Returns pts3d (B, H, W, 3) of the reference view in its own camera frame and conf (B, H, W)."""
        out = self.head(self.decode(f_ref, f_src, pos_ref, pos_src), H // self.patch, W // self.patch).float()
        xyz = out[:, :3].permute(0, 2, 3, 1)
        d = xyz.norm(dim=-1, keepdim=True)
        pts = xyz / d.clamp_min(1e-8) * torch.expm1(d)
        return pts, 1.0 + out[:, 3].exp()

    def forward(self, img_ref, img_src):
        (f1, p1), (f2, p2) = self.encode(img_ref), self.encode(img_src)
        return self.pointmap(f1, f2, p1, p2, img_ref.shape[2], img_ref.shape[3])
