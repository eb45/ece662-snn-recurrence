#!/bin/bash
# Submit the W_rec learning-rate x placement grid plus anchored-decay runs.
#
#   DRY_RUN=1 bash sweep_recurrent.sh poc    # print what would be submitted
#   bash sweep_recurrent.sh poc              # seed 2026 only (proof of concept)
#   bash sweep_recurrent.sh full             # seeds 2026, 1, 2
#   bash sweep_recurrent.sh seeds            # seeds 1, 2: fc arms B-E + control A
#
# Runs whose folder already has final.json are skipped, so the seed-2026 runs
# you already have (rec_fc_s2026, rec_conv5_fc_s2026, rec_fc_rlr1e-4_s2026,
# rec_conv5_fc_frozen_s2026) are not repeated, and re-running this after a
# partial sweep only submits what is missing. Unfinished runs that are already
# queued would be submitted twice, so check squeue before re-running.
#
# Env: RUNS_ROOT (default /work/$USER/ece662_runs), TIME (default 02:00:00),
# plus anything train_recurrent.sbatch reads (CONDA_ENV, DATA_ROOT, ...).

set -euo pipefail
MODE="${1:-poc}"
EXTRA_SEEDS=""
case "$MODE" in
  poc)   SEEDS="2026" ;;
  full)  SEEDS="2026 1 2" ;;
  seeds) SEEDS=""; EXTRA_SEEDS="1 2" ;;
  *)     echo "usage: bash sweep_recurrent.sh poc|full|seeds"; exit 1 ;;
esac
RUNS_ROOT="${RUNS_ROOT:-/work/${USER}/ece662_runs}"
TIME="${TIME:-02:00:00}"
# DVS-GC: DATASET, T, and DVSGC_ARGS pass through to train_recurrent.sbatch; run
# names get the same gc81p_ / gc96u_ prefix it uses.
DS_TAG=""
if [ "${DATASET:-dvs128}" = "dvsgc" ]; then
  case "${DVSGC_ARGS:-}" in *--no-repeat*) DS_TAG="gc96u_" ;; *) DS_TAG="gc81p_" ;; esac
fi
n_submit=0

# submit LAYERS SEED FROZEN REC_LR REC_WD ANCHOR
# REC_LR "" means W_rec uses the network lr (1e-3). The name follows the
# convention in train_recurrent.sbatch so existing runs are recognized.
submit() {
  local layers="$1" seed="$2" frozen="$3" rlr="$4" wd="$5" anchor="$6"
  local name="${DS_TAG}rec_${layers// /_}"
  [ "$frozen" = 1 ] && name="${name}_frozen"
  [ -n "$rlr" ] && name="${name}_rlr${rlr}"
  [ "$wd" != 0 ] && name="${name}_wd${wd}"
  [ "$anchor" = 1 ] && name="${name}_anchor"
  name="${name}_s${seed}"

  if [ -f "$RUNS_ROOT/$name/final.json" ]; then
    echo "skip   $name (finished)"
    return
  fi
  echo "submit $name"
  n_submit=$((n_submit + 1))
  [ "${DRY_RUN:-0}" = 1 ] && return
  REC_LAYERS="$layers" SEED="$seed" FREEZE_REC="$frozen" REC_LR="$rlr" REC_WD="$wd" \
    REC_ANCHOR="$anchor" RUN_NAME="$name" OUT="$RUNS_ROOT/$name" DS_TAG="$DS_TAG" \
    sbatch --job-name="$name" --time="$TIME" train_recurrent.sbatch
}

# submit_control SEED: arm A (feed-forward control). DVS128 uses train_control.sbatch;
# DVS-GC uses train_extra.sbatch with the same DATASET, T and DVSGC_ARGS as the
# recurrent runs, named like Emma's seed-2026 run (dvsgc96u_A_s2026).
submit_control() {
  local seed="$1" name="control_s$1" script="train_control.sbatch"
  if [ -n "$DS_TAG" ]; then
    name="dvs${DS_TAG%_}_A_s${seed}"
    script="train_extra.sbatch"
  fi
  if [ -f "$RUNS_ROOT/$name/final.json" ]; then
    echo "skip   $name (finished)"
    return
  fi
  echo "submit $name"
  n_submit=$((n_submit + 1))
  [ "${DRY_RUN:-0}" = 1 ] && return
  SEED="$seed" RUN_NAME="$name" OUT="$RUNS_ROOT/$name" \
    DATA_ROOT="${DATA_ROOT:-/work/${USER}/dvs128_gesture}" \
    sbatch --job-name="$name" --time="$TIME" "$script"
}

for seed in $SEEDS; do
  for layers in "fc" "conv5" "conv5 fc"; do
    # Frozen random W_rec: the zero-drift end of every curve.
    submit "$layers" "$seed" 1 "" 0 0
    # W_rec lr grid ("" = 1e-3, the network lr).
    for rlr in "" 3e-4 1e-4 3e-5; do
      submit "$layers" "$seed" 0 "$rlr" 0 0
    done
  done

  # Anchored decay (toward W0) vs. plain decay (toward 0), at the network lr.
  # AdamW-style decay shrinks dW by about exp(-2.35 * wd) over 64 epochs here,
  # so wd 0.3 / 1 / 3 keep roughly 50% / 10% / 0.1% of an unused direction.
  for wd in 0.3 1.0 3.0; do
    submit "fc" "$seed" 0 "" "$wd" 1
  done
  submit "fc" "$seed" 0 "" 1.0 0
  submit "conv5" "$seed" 0 "" 1.0 1
  submit "conv5 fc" "$seed" 0 "" 1.0 1
done

# Extra seeds for the paper arms only: fc B (frozen), C (lr 1e-3), D (lr 1e-4),
# E (anchored wd 1), plus control A. The grid table is seed 2026 only.
for seed in $EXTRA_SEEDS; do
  submit "fc" "$seed" 1 "" 0 0
  submit "fc" "$seed" 0 "" 0 0
  submit "fc" "$seed" 0 1e-4 0 0
  submit "fc" "$seed" 0 "" 1.0 1
  submit_control "$seed"
done

echo "$n_submit job(s) $([ "${DRY_RUN:-0}" = 1 ] && echo 'would be ' || true)submitted"
