"""Load public pretrained pointmap weights (DUSt3R ViT-L/ViT-B 512 DPT) into PointmapNet.

Only the tensors the pruned network needs are kept: the second DPT head, the
source decoder's last block, CroCo's unused mask token and the duplicated
`layer_rn` aliases are skipped.
"""
import os
import re
import urllib.request

import torch

from ..common.io import log
from .network import PointmapNet

DEFAULT_URL = "https://huggingface.co/naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt/resolve/main/model.safetensors"
DEFAULT_NAME = "DUSt3R_ViTLarge_BaseDecoder_512_dpt.safetensors"


def download(dest: str, url: str = DEFAULT_URL) -> str:
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    log(f"downloading pointmap weights -> {dest}")
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def _read(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("model", ckpt)


def remap(sd: dict, dec_depth: int = 12):
    out, skipped = {}, 0
    for k, v in sd.items():
        if k.startswith("downstream_head2.") or k == "mask_token" or ".layer_rn." in k:
            skipped += v.numel()
            continue
        m = re.match(r"dec_blocks2\.(\d+)\.(.*)", k)
        if m:
            if int(m.group(1)) >= dec_depth - 1:
                skipped += v.numel()
                continue
            k = f"src_blocks.{m.group(1)}.{m.group(2)}"
        k = re.sub(r"^dec_blocks\.", "ref_blocks.", k)
        k = k.replace("patch_embed.proj.", "patch_embed.")
        if k.startswith("downstream_head1.dpt."):
            k = k[len("downstream_head1.dpt."):]
            k = re.sub(r"^scratch\.layer(\d)_rn\.", lambda g: f"layer_rn.{int(g.group(1)) - 1}.", k)
            k = re.sub(r"^scratch\.refinenet(\d)\.", lambda g: f"refinenet.{int(g.group(1)) - 1}.", k)
            k = k.replace("resConfUnit1.", "res1.").replace("resConfUnit2.", "res2.")
            k = "head." + re.sub(r"^head\.", "out.", k)
        out[k] = v
    return out, skipped


def load_pointmap_net(path: str, device="cpu") -> PointmapNet:
    sd, skipped = remap(_read(path))
    net = PointmapNet()
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    n = sum(p.numel() for p in net.parameters())
    log(f"pointmap net: {n / 1e6:.1f}M parameters loaded, {skipped / 1e6:.1f}M redundant skipped")
    return net.to(device).eval()
