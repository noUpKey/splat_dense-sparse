#!/usr/bin/env bash
#SBATCH --job-name=uwsplat
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Full pipeline on all four Submerged3D scenes + ablations. Adjust --gres / partition to your cluster.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
eval "$(conda shell.bash hook)" 2>/dev/null && conda activate uwsplat || source .venv/bin/activate
python scripts/run_all.py --data Submerged3D --out outputs \
       --variants full,sparse_init,dense_init,no_medium,no_depth_priors,baseline_3dgs
