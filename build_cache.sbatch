#!/bin/bash
#SBATCH --job-name=dvs_cache
#SBATCH --output=logs/dvs_cache_%j.out
#SBATCH --error=logs/dvs_cache_%j.err
#SBATCH --account=naderilab
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

# Folder containing dvs_data.py. Defaults to wherever you ran sbatch.
# If you submit from somewhere else, replace this with the full path to your 662 folder.
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

ENV_ROOT="/hpc/group/naderilab/eb408/peft_proteomics"
DATA_ROOT="/work/eb408/dvs128_gesture"

cd "$PROJECT_DIR" || exit 1
source "$ENV_ROOT/peft_proteomics_env/bin/activate"

nvidia-smi
python dvs_data.py --root "$DATA_ROOT"