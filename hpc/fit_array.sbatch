#!/bin/bash
# One array task = one hyperparameter set (one row of `python -m gectargets
# tasks CONFIG`); inside it, subjects are fitted in parallel on the task's CPUs.
# Submit with hpc/submit.sh, which computes the targets first and sets the
# array size. Edit the #SBATCH lines and ENV for your cluster.

#SBATCH --job-name=gec_fit
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out

set -euo pipefail
: "${CONFIG:?set CONFIG (exported by submit.sh)}"
ENV="${ENV:-gectargets}"   # conda env with the package installed

eval "$(conda shell.bash hook)"
conda activate "$ENV"
# One BLAS thread per process: the parallelism is over subjects
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u -m gectargets fit "$CONFIG" \
    --task "$SLURM_ARRAY_TASK_ID" --n-jobs "$SLURM_CPUS_PER_TASK"
