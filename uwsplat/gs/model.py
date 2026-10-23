"""Gaussian parameters (gsplat ParameterDict layout), initialisation and PLY export."""
import numpy as np
import torch
from scipy.spatial import cKDTree

from ..common.io import write_ply
from .sh import rgb_to_sh

SPARSE, DENSE = 0, 1  # modality tags: SIFT/wreck points, dense/water points


def init_params(xyz: np.ndarray, rgb: np.ndarray, opacity: np.ndarray, sh_degree: int, device) -> torch.nn.ParameterDict:
    """xyz (N, 3), rgb (N, 3) restored colours in [0, 1], opacity (N,) initial opacities."""
    d, _ = cKDTree(xyz).query(xyz, k=4)
    dist = np.sqrt(np.mean(d[:, 1:] ** 2, 1)).clip(1e-7)  # 3DGS: RMS distance to the 3 nearest neighbours
    n = len(xyz)
    p = {
        "means": torch.tensor(xyz, dtype=torch.float32),
        "scales": torch.log(torch.tensor(dist, dtype=torch.float32))[:, None].repeat(1, 3),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        "opacities": torch.logit(torch.tensor(opacity, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)),
        "sh0": rgb_to_sh(torch.tensor(rgb, dtype=torch.float32))[:, None],
    }
    if sh_degree > 0:
        p["shN"] = torch.zeros(n, (sh_degree + 1) ** 2 - 1, 3)
    return torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in p.items()}).to(device)


def sh_coeffs(params) -> torch.Tensor:
    return torch.cat([params["sh0"], params["shN"]], 1) if "shN" in params else params["sh0"]


@torch.no_grad()
def export_ply(params, path: str):
    """Standard 3DGS PLY (opens in common viewers). Colours are the *restored* radiance J."""
    xyz = params["means"].cpu().numpy()
    fields = {k: np.zeros(len(xyz), np.float32) for k in ("nx", "ny", "nz")}
    dc = params["sh0"][:, 0].cpu().numpy()
    for c in range(3):
        fields[f"f_dc_{c}"] = dc[:, c]
    if "shN" in params:
        rest = params["shN"].transpose(1, 2).reshape(len(xyz), -1).cpu().numpy()
        for k in range(rest.shape[1]):
            fields[f"f_rest_{k}"] = rest[:, k]
    fields["opacity"] = params["opacities"].cpu().numpy()
    for k in range(3):
        fields[f"scale_{k}"] = params["scales"][:, k].cpu().numpy()
    for k in range(4):
        fields[f"rot_{k}"] = params["quats"][:, k].cpu().numpy()
    write_ply(path, xyz, None, **{k: v.astype(np.float32) for k, v in fields.items()})
