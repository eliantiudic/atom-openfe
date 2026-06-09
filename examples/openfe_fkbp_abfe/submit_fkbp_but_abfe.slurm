#!/bin/bash
#SBATCH --job-name=atom-fkbp-but
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/users/etiudic/atom-openfe-integration}"
CONDA_ENV="${CONDA_ENV:-openfe-cuda124}"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/atom-openfe/examples/openfe_fkbp_abfe/runs/fkbp_but_atom_abfe}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
WALL_TIME_MINUTES="${WALL_TIME_MINUTES:-1200}"
PRODUCTION_STEPS="${PRODUCTION_STEPS:-10000}"

mkdir -p "${REPO_ROOT}/atom-openfe/examples/openfe_fkbp_abfe/logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${REPO_ROOT}/atom-openfe/src:${REPO_ROOT}/AToM-OpenMM:${PYTHONPATH:-}"
export MPLCONFIGDIR="${WORK_DIR}/mplconfig"
mkdir -p "${MPLCONFIGDIR}"

python "${REPO_ROOT}/atom-openfe/examples/openfe_fkbp_abfe/run_fkbp_but_abfe.py" \
  --work-dir "${WORK_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --wall-time-minutes "${WALL_TIME_MINUTES}" \
  --production-steps "${PRODUCTION_STEPS}"
