#!/bin/bash
#SBATCH --job-name=lsm_train
#SBATCH --output=logs/lsm_%j.out
#SBATCH --error=logs/lsm_%j.err
#SBATCH --account=naderilab
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --requeue

EPOCHS="${EPOCHS:-64}"
SEED="${SEED:-2026}"
RUN_NAME="${RUN_NAME:-lsm_s${SEED}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUM="${ACCUM:-2}"
LIF_NEURONS="${LIF_NEURONS:-2794094}"

DATA_ROOT="${DATA_ROOT:-/work/eb408/dvs128_gesture}"
OUT="${OUT:-/work/eb408/ece662_runs/${RUN_NAME}}"

cd "$SLURM_SUBMIT_DIR" || exit 1
source /hpc/group/naderilab/eb408/peft_proteomics/peft_proteomics_env/bin/activate

mkdir -p logs "$OUT"
nvidia-smi
echo "run: $RUN_NAME  out: $OUT  epochs: $EPOCHS  seed: $SEED"

python -u train_lsm.py \
  --root "$DATA_ROOT" \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --batch-size "$BATCH_SIZE" \
  --accum "$ACCUM" \
  --lif-neurons "$LIF_NEURONS"
