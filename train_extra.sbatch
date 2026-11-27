#!/bin/bash
#SBATCH --job-name=ctrl_extra
#SBATCH --output=logs/extra_%j.out
#SBATCH --error=logs/extra_%j.err
#SBATCH --account=naderilab
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --requeue

# Feed-forward control on an extra dataset (copy of train_control.sbatch that
# runs in the same venv and passes --dataset through).
#
# Debug run:   DATASET=dvsgc EPOCHS=2 RUN_NAME=dvsgc_debug sbatch --time=01:00:00 train_extra.sbatch
# 96-u:        DATASET=dvsgc T=32 RUN_NAME=dvsgc96u_A_s2026 \
#              DVSGC_ARGS="--seq-len 6 --alpha-min 0.2 --alpha-max 1.0 --no-repeat" sbatch train_extra.sbatch
#              (always set RUN_NAME: the default name is the same for 81-p and 96-u)
# IF neurons:  NEURON=if RUN_NAME=dvsgc96u_A_if_s2026 T=32 DVSGC_ARGS="..." sbatch train_extra.sbatch
# Short run:   DATASET=dvsgc EPOCHS=20 RUN_NAME=dvsgc_ff_s2026 sbatch train_extra.sbatch
#
# Build the dataset cache first (build_extra_cache.sbatch), otherwise the
# first job spends its time building it. If the job is preempted it requeues
# and train_control.py resumes from <OUT>/checkpoint.pt, so keep RUN_NAME the
# same across restarts.

DATASET="${DATASET:-dvsgc}"
EPOCHS="${EPOCHS:-64}"
SEED="${SEED:-2026}"
T="${T:-16}"
RUN_NAME="${RUN_NAME:-${DATASET}_ff_s${SEED}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUM="${ACCUM:-2}"
VENV="${VENV:-/hpc/group/naderilab/eb408/peft_proteomics/peft_proteomics_env}"

DATA_ROOT="${DATA_ROOT:-/work/eb408/dvs128_gesture}"
OUT="${OUT:-/work/eb408/ece662_runs/${RUN_NAME}}"

cd "$SLURM_SUBMIT_DIR" || exit 1
if [ -n "${CONDA_ENV:-}" ]; then
  # sbatch inherits PATH from the submitting shell, so conda is found if it works there.
  source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "conda not found"; exit 1; }
  conda activate "$CONDA_ENV" || exit 1
else
  source "$VENV/bin/activate" || { echo "could not activate $VENV"; exit 1; }
fi

mkdir -p logs "$OUT"
nvidia-smi
echo "dataset: $DATASET  run: $RUN_NAME  out: $OUT  epochs: $EPOCHS  seed: $SEED  T: $T"

python -u train_control.py \
  --dataset "$DATASET" \
  --root "$DATA_ROOT" \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --T "$T" \
  --batch-size "$BATCH_SIZE" \
  --accum "$ACCUM" \
  --neuron "${NEURON:-lif}" \
  ${DVSGC_ARGS:-}
