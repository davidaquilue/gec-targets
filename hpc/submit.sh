#!/bin/bash
# Usage: bash hpc/submit.sh CONFIG [ENV]
# Submits the targets job and, when it succeeds, one array task per fit task.
# Resubmitting is safe: finished targets and subjects are skipped.
set -euo pipefail
export CONFIG="$(realpath "$1")"
export ENV="${2:-gectargets}"
mkdir -p slurm_logs

eval "$(conda shell.bash hook)"
conda activate "$ENV"
N=$(python -c "from gectargets.config import load_config, expand_tasks; \
print(len(expand_tasks(load_config('$CONFIG'))))")
echo "$N fit tasks"

DIR="$(dirname "$0")"
JID=$(sbatch --parsable --export=ALL "$DIR/targets.sbatch")
sbatch --dependency=afterok:"$JID" --array=0-$((N - 1)) --export=ALL \
    "$DIR/fit_array.sbatch"
