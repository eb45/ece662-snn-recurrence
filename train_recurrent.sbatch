#!/bin/bash
#SBATCH --job-name=rec_train
#SBATCH --output=logs/recurrent_%j.out
#SBATCH --error=logs/recurrent_%j.err
#SBATCH --account=naderilab
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --requeue

# Debug run:        EPOCHS=2 RUN_NAME=rec_debug sbatch --time=00:45:00 train_recurrent.sbatch
# Main arm:         sbatch train_recurrent.sbatch                     (conv5 + fc, learned)
# FC only:          REC_LAYERS="fc" sbatch train_recurrent.sbatch
# Random recurrence: FREEZE_REC=1 sbatch train_recurrent.sbatch
# Late readout:     READOUT=late sbatch train_recurrent.sbatch
#
# If the job is preempted it requeues and train_recurrent.py resumes from
# <OUT>/checkpoint.pt, so keep RUN_NAME the same across restarts.

EPOCHS="${EPOCHS:-64}"
SEED="${SEED:-2026}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUM="${ACCUM:-2}"
REC_LAYERS="${REC_LAYERS:-conv5 fc}"
REC_INIT="${REC_INIT:-orthogonal}"
REC_GAIN="${REC_GAIN:-0.5}"
FREEZE_REC="${FREEZE_REC:-0}"
READOUT="${READOUT:-mean}"
LATE_K="${LATE_K:-4}"
CLIP_GRAD="${CLIP_GRAD:-0}"

# Default run name encodes the variant, e.g. rec_conv5_fc_frozen_late_s2026
TAG="rec_${REC_LAYERS// /_}"
[ "$FREEZE_REC" = "1" ] && TAG="${TAG}_frozen"
[ "$READOUT" = "late" ] && TAG="${TAG}_late"
RUN_NAME="${RUN_NAME:-${TAG}_s${SEED}}"

DATA_ROOT="${DATA_ROOT:-/work/eb408/dvs128_gesture}"
OUT="${OUT:-/work/eb408/ece662_runs/${RUN_NAME}}"

EXTRA=()
[ "$FREEZE_REC" = "1" ] && EXTRA+=(--freeze-rec)

cd "$SLURM_SUBMIT_DIR" || exit 1
source /hpc/group/naderilab/eb408/peft_proteomics/peft_proteomics_env/bin/activate

mkdir -p logs "$OUT"
nvidia-smi
echo "run: $RUN_NAME  out: $OUT  epochs: $EPOCHS  seed: $SEED  rec: $REC_LAYERS  freeze: $FREEZE_REC  readout: $READOUT"

# REC_LAYERS is deliberately unquoted so "conv5 fc" becomes two arguments.
python -u train_recurrent.py \
  --root "$DATA_ROOT" \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --batch-size "$BATCH_SIZE" \
  --accum "$ACCUM" \
  --rec-layers $REC_LAYERS \
  --rec-init "$REC_INIT" \
  --rec-gain "$REC_GAIN" \
  --readout "$READOUT" \
  --late-k "$LATE_K" \
  --clip-grad "$CLIP_GRAD" \
  "${EXTRA[@]}"
