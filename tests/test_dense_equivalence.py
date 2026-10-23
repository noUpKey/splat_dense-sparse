"""Numerical equivalence of uwsplat's pruned PointmapNet with the official DUSt3R model.

python -m tests.test_dense_equivalence <path/to/dust3r_repo> <checkpoint> <img1> <img2>

Both models load the same checkpoint and predict the reference view's pointmap
and confidence for a real image pair; the outputs must agree to float precision.
"""
import sys

import numpy as np
import torch
from PIL import Image

from uwsplat.dense.weights import _read, load_pointmap_net


def load(path, w=512, h=336):
    im = Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS)
    return torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float()[None] / 127.5 - 1.0


def main(repo, ckpt, a, b):
    sys.path.insert(0, repo)
    from dust3r.model import AsymmetricCroCo3DStereo
    inf = float("inf")
    ref = AsymmetricCroCo3DStereo(
        output_mode="pts3d", head_type="dpt", depth_mode=("exp", -inf, inf), conf_mode=("exp", 1, inf),
        enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12,
        img_size=(512, 512), patch_embed_cls="ManyAR_PatchEmbed", pos_embed="RoPE100", landscape_only=False)
    sd = _read(ckpt)
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    assert all(".layer_rn." in k for k in missing) and not unexpected, (missing, unexpected)
    ref.eval()
    ours = load_pointmap_net(ckpt)
    x1, x2 = load(a), load(b)
    with torch.no_grad():
        shape = torch.tensor([[x1.shape[2], x1.shape[3]]])
        r1, _ = ref(dict(img=x1, true_shape=shape, idx=[0], instance=["0"]),
                    dict(img=x2, true_shape=shape, idx=[1], instance=["1"]))
        pts, conf = ours(x1, x2)
    dp = (pts - r1["pts3d"]).abs().max().item()
    dc = ((conf - r1["conf"]).abs() / r1["conf"]).max().item()
    print(f"pts3d range [{r1['pts3d'].min():.3f}, {r1['pts3d'].max():.3f}]  max |diff| {dp:.2e}")
    print(f"conf  range [{r1['conf'].min():.3f}, {r1['conf'].max():.3f}]  max rel diff {dc:.2e}")
    assert dp < 1e-3 and dc < 1e-3
    print("dense network matches the reference implementation")


if __name__ == "__main__":
    main(*sys.argv[1:5])
