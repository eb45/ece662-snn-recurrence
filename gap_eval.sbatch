#!/bin/bash
#SBATCH --job-name=gap_eval
#SBATCH --output=logs/gap_eval_%j.out
#SBATCH --error=logs/gap_eval_%j.err
#SBATCH --account=naderilab
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --requeue

# Gap, reverse, and shuffle tests on finished runs. Pass run folders as arguments:
#   sbatch gap_eval.sbatch /work/$USER/ece662_runs/control_s2026 /work/$USER/ece662_runs/rec_fc_s2026
# Or every finished run in a folder:
#   sbatch gap_eval.sbatch /work/$USER/ece662_runs/*/
# Extra flags go in EXTRA, e.g. EXTRA="--bn-check" or EXTRA="--which best".
# Runs whose folder already has gap.json are skipped, so a requeued job picks up where it stopped.

VENV="${VENV:-/hpc/group/naderilab/eb408/peft_proteomics/peft_proteomics_env}"
cd "$SLURM_SUBMIT_DIR" || exit 1
if [ -n "${CONDA_ENV:-}" ]; then
  # sbatch inherits PATH from the submitting shell, so conda is found if it works there.
  source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "conda not found"; exit 1; }
  conda activate "$CONDA_ENV" || exit 1
else
  source "$VENV/bin/activate" || { echo "could not activate $VENV"; exit 1; }
fi
mkdir -p logs

todo=()
for run in "$@"; do
  if [ ! -f "$run/final.json" ]; then echo "skip $run (not finished)"; continue; fi
  if [ -f "$run/gap.json" ]; then echo "skip $run (gap.json exists)"; continue; fi
  todo+=("$run")
done
[ ${#todo[@]} -eq 0 ] && { echo "nothing to do"; exit 0; }

python -u gap_eval.py --runs "${todo[@]}" ${EXTRA:-}
