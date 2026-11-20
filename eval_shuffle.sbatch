#!/bin/bash
#SBATCH --job-name=shuffle_eval
#SBATCH --output=logs/shuffle_%j.out
#SBATCH --error=logs/shuffle_%j.err
#SBATCH --account=naderilab
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --requeue

# Time-shuffle test on a finished run. Writes <RUN_DIR>/shuffle.json.
#   RUN_NAME=dvsgc_ff_s2026 sbatch eval_shuffle.sbatch

RUN_NAME="${RUN_NAME:?set RUN_NAME to the finished run folder name}"
RUN_DIR="${RUN_DIR:-/work/eb408/ece662_runs/${RUN_NAME}}"
VENV="${VENV:-/hpc/group/naderilab/eb408/peft_proteomics/peft_proteomics_env}"

cd "$SLURM_SUBMIT_DIR" || exit 1
source "$VENV/bin/activate" || { echo "could not activate $VENV"; exit 1; }

mkdir -p logs
nvidia-smi
python -u eval_shuffle.py --run "$RUN_DIR"
