#!/bin/bash
# Empirical targets of every (dataset, signal variant) in the sweep. Cheap
# compared with the fits; run once before the fit array (submit.sh does it).

#SBATCH --job-name=gec_targets
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out

set -euo pipefail
: "${CONFIG:?set CONFIG (exported by submit.sh)}"
ENV="${ENV:-gectargets}"

eval "$(conda shell.bash hook)"
conda activate "$ENV"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u -m gectargets targets "$CONFIG" --n-jobs "$SLURM_CPUS_PER_TASK"
