#!/bin/bash
#SBATCH --job-name=atom-fkbp-abfe
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -euo pipefail

EXAMPLE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ATOM_OPENFE_ROOT="${ATOM_OPENFE_ROOT:-$(cd "${EXAMPLE_DIR}/../../.." && pwd)}"
ATOM_OPENMM_ROOT="${ATOM_OPENMM_ROOT:-${ATOM_OPENFE_ROOT}/../AToM-OpenMM}"
CONDA_ENV="${CONDA_ENV:-openfe-cuda124}"
LIGAND="${LIGAND:-but}"
WORK_DIR="${WORK_DIR:-${EXAMPLE_DIR}/runs}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
WALL_TIME_MINUTES="${WALL_TIME_MINUTES:-1200}"
PRODUCTION_STEPS="${PRODUCTION_STEPS:-10000}"

mkdir -p "${EXAMPLE_DIR}/logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${ATOM_OPENFE_ROOT}/src:${PYTHONPATH:-}"
if [ -d "${ATOM_OPENMM_ROOT}/atom_openmm" ]; then
  export PYTHONPATH="${ATOM_OPENMM_ROOT}:${PYTHONPATH}"
fi

python "${EXAMPLE_DIR}/run.py" \
  --ligand "${LIGAND}" \
  --work-dir "${WORK_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --wall-time-minutes "${WALL_TIME_MINUTES}" \
  --production-steps "${PRODUCTION_STEPS}"
