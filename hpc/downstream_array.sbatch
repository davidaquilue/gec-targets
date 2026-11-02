#!/bin/bash
# Downstream measures (FDT, EPR) of the fits: one array task per fit task,
# same ids as `python -m gectargets tasks CONFIG`; inside it, subjects run in
# parallel on the task's CPUs. Needs the fits. Submit with
#     CONFIG=$(realpath CONFIG) ENV=gectargets sbatch --export=ALL \
#         --array=0-$((N - 1)) hpc/downstream_array.sbatch
# (N: number of tasks). Resubmitting is safe: finished subjects are skipped.

#SBATCH --job-name=gec_downstream
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out

set -euo pipefail
: "${CONFIG:?set CONFIG}"
ENV="${ENV:-gectargets}"   # conda env with the package installed

eval "$(conda shell.bash hook)"
conda activate "$ENV"
# One BLAS thread per process: the parallelism is over subjects
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u -m gectargets downstream "$CONFIG" \
    --task "$SLURM_ARRAY_TASK_ID" --n-jobs "$SLURM_CPUS_PER_TASK"
