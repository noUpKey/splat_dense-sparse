"""Gaussian parameters (gsplat ParameterDict layout), initialisation and PLY export."""
import numpy as np
import torch
from scipy.spatial import cKDTree

from ..common.io import write_ply
from .sh import rgb_to_sh

SPARSE, DENSE, VERIFIED = 0, 1, 2  # modality tags: SIFT points, network-only (water/completed), MVS-verified


def normal_to_quat(n: np.ndarray) -> np.ndarray:
    """Unit normals (N, 3) -> quaternions (w, x, y, z) of the shortest rotation taking the local z axis
    onto each normal (gsplat convention: covariance = R diag(s)^2 R^T, so scale[2] lies along n)."""
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    q = np.stack([1.0 + n[:, 2], -n[:, 1], n[:, 0], np.zeros(len(n))], 1)
    flip = q[:, 0] < 1e-6  # n ~ -z: rotate by pi about x
    q[flip] = np.array([0.0, 1.0, 0.0, 0.0])
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def init_params(xyz: np.ndarray, rgb: np.ndarray, opacity: np.ndarray, sh_degree: int, device,
                normals: np.ndarray = None, surfel_ratio: float = 0.0) -> torch.nn.ParameterDict:
    """xyz (N, 3), rgb (N, 3) restored colours in [0, 1], opacity (N,) initial opacities.

    Scales follow 3DGS (RMS distance to the 3 nearest neighbours). With surfel_ratio > 0, points that
    carry a surface normal (|n| > 0.5) start as flat discs on the surface: the in-plane scales keep the
    neighbour distance, the scale along the normal is surfel_ratio times it."""
    d, _ = cKDTree(xyz).query(xyz, k=4)
    dist = np.sqrt(np.mean(d[:, 1:] ** 2, 1)).clip(1e-7)  # 3DGS: RMS distance to the 3 nearest neighbours
    n = len(xyz)
    scales = np.repeat(dist[:, None], 3, 1)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    if normals is not None and surfel_ratio > 0:
        has = np.linalg.norm(normals, axis=1) > 0.5
        quats[has] = normal_to_quat(normals[has])
        scales[has, 2] *= surfel_ratio
    p = {
        "means": torch.tensor(xyz, dtype=torch.float32),
        "scales": torch.log(torch.tensor(scales, dtype=torch.float32)),
        "quats": torch.tensor(quats, dtype=torch.float32),
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
