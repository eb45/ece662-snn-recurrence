#!/bin/bash
#SBATCH --job-name=extra_cache
#SBATCH --output=logs/extra_cache_%j.out
#SBATCH --error=logs/extra_cache_%j.err
#SBATCH --account=naderilab
#SBATCH --partition=common
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# Build an extra dataset's cache and sanity check it. CPU only.
# Runs on the common partition (not scavenger) so a preemption cannot leave a
# half-written cache file behind.
#
#   DATASET=dvsgc sbatch build_extra_cache.sbatch
#   96-u at T=32:
#   T=32 DVSGC_ARGS="--seq-len 6 --alpha-min 0.2 --alpha-max 1.0 --no-repeat" sbatch build_extra_cache.sbatch

DATASET="${DATASET:-dvsgc}"
T="${T:-16}"
VENV="${VENV:-/hpc/group/naderilab/eb408/peft_proteomics/peft_proteomics_env}"
DATA_ROOT="${DATA_ROOT:-/work/eb408/dvs128_gesture}"

cd "$SLURM_SUBMIT_DIR" || exit 1
source "$VENV/bin/activate" || { echo "could not activate $VENV"; exit 1; }

mkdir -p logs
echo "dataset: $DATASET  T: $T  root: $DATA_ROOT"
time python -u datasets_extra.py --dataset "$DATASET" --root "$DATA_ROOT" --T "$T" ${DVSGC_ARGS:-}
