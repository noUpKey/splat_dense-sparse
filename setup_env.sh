#!/usr/bin/env bash
# Environment for a Linux GPU node (tested target: NVIDIA A40, sm_86).
#   bash setup_env.sh                 # conda env "uwsplat" if conda exists, else ./.venv
#   CUDA_TAG=cu118 bash setup_env.sh  # for drivers older than CUDA 12.1
# Prebuilt gsplat 1.5.3 wheels exist for Python 3.10 + torch 2.4 (cu118/cu121/cu124) and
# contain sm_86 kernels, so nothing has to be compiled.
set -euo pipefail
cd "$(dirname "$0")"
ENV_NAME=${ENV_NAME:-uwsplat}
CUDA_TAG=${CUDA_TAG:-cu121}

if command -v conda >/dev/null 2>&1; then
  conda create -y -n "$ENV_NAME" python=3.10
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
else
  python3.10 -m venv .venv
  source .venv/bin/activate
fi

pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install "gsplat==1.5.3+pt24${CUDA_TAG}" --find-links https://docs.gsplat.studio/whl/gsplat/
pip install -r requirements.txt

python scripts/download_weights.py                        # pointmap network weights (2.3 GB)
python -c "import lpips; lpips.LPIPS(net='vgg')"          # caches VGG16 for LPIPS (compute nodes may lack internet)
python - <<'EOF'
import torch, gsplat
from gsplat import rasterization
assert torch.cuda.is_available(), "CUDA not available"
dev = "cuda"
m = torch.rand(100, 3, device=dev) + torch.tensor([0, 0, 2.0], device=dev)
q = torch.randn(100, 4, device=dev); s = torch.full((100, 3), 0.05, device=dev)
o = torch.full((100,), 0.5, device=dev); c = torch.rand(1, 100, 6, device=dev)
K = torch.tensor([[[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]]], device=dev)
img, alpha, _ = rasterization(m, q, s, o, c, torch.eye(4, device=dev)[None], K, 320, 240, render_mode="RGB+ED", packed=False)
print("OK torch", torch.__version__, "| gsplat", gsplat.__version__, "|", torch.cuda.get_device_name(0), "| render", tuple(img.shape))
EOF
