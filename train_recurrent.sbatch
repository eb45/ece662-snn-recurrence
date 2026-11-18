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
# Regularize W_rec: REC_LAYERS="fc" REC_WD=1.0 sbatch train_recurrent.sbatch   (also REC_LR=1e-4)
#
# Python env: CONDA_ENV=<name> activates a conda env; otherwise VENV=<path> (default:
# Emma's shared virtualenv) is sourced. Data and outputs default to /work/$USER.
#   CONDA_ENV=snn sbatch train_recurrent.sbatch
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
REC_LR="${REC_LR:-}"
REC_WD="${REC_WD:-0}"

# Default run name encodes every non-default setting, e.g. rec_fc_zero_wd1.0_s2026,
# so two variants never share an output folder (and a checkpoint).
TAG="rec_${REC_LAYERS// /_}"
[ "$REC_INIT" = "zero" ] && TAG="${TAG}_zero"
[ "$REC_GAIN" != "0.5" ] && TAG="${TAG}_g${REC_GAIN}"
[ "$FREEZE_REC" = "1" ] && TAG="${TAG}_frozen"
[ "$READOUT" = "late" ] && TAG="${TAG}_late"
[ "$CLIP_GRAD" != "0" ] && TAG="${TAG}_clip${CLIP_GRAD}"
[ -n "$REC_LR" ] && TAG="${TAG}_rlr${REC_LR}"
[ "$REC_WD" != "0" ] && TAG="${TAG}_wd${REC_WD}"
RUN_NAME="${RUN_NAME:-${TAG}_s${SEED}}"

DATA_ROOT="${DATA_ROOT:-/work/${USER}/dvs128_gesture}"
OUT="${OUT:-/work/${USER}/ece662_runs/${RUN_NAME}}"

EXTRA=()
[ "$FREEZE_REC" = "1" ] && EXTRA+=(--freeze-rec)
[ -n "$REC_LR" ] && EXTRA+=(--rec-lr "$REC_LR")

cd "$SLURM_SUBMIT_DIR" || exit 1
if [ -n "${CONDA_ENV:-}" ]; then
  # sbatch inherits PATH from the submitting shell, so conda is found if it works there.
  source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "conda not found"; exit 1; }
  conda activate "$CONDA_ENV" || exit 1
else
  source "${VENV:-/hpc/group/naderilab/eb408/peft_proteomics/peft_proteomics_env}/bin/activate" || exit 1
fi

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
  --rec-wd "$REC_WD" \
  "${EXTRA[@]}"
