#!/usr/bin/env bash
# Data-scarcity sweep for the FedMeta-vs-Meta-at-low-data hypothesis (and R1.2
# minimum-traffic-volume). Trains meta and federated at several data fractions
# and seeds, snapshotting each run's checkpoints so the annotation-F1 eval can
# score them afterward. Rei runs this (produces paper numbers). ~7h for the full
# grid; it is resumable (skips runs whose snapshot already exists).
#
#   bash scripts/scarcity_sweep.sh
#
# Trims: edit FRACTIONS / SEEDS below to shrink the grid.
set -euo pipefail
cd "$(dirname "$0")/.."

FRACTIONS=(1.0 0.5 0.25 0.1)
SEEDS=(42 43 44)
MODELS=(meta federated)
SNAP=results/_scarcity
COMMON="--T 60 --is_save --use_historical_data --skip_continuous_learning"

mkdir -p "$SNAP"
for frac in "${FRACTIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for model in "${MODELS[@]}"; do
      dst="$SNAP/${model}_f${frac}_s${seed}"
      if [ -d "$dst" ] && ls "$dst"/*.pth >/dev/null 2>&1; then
        echo "SKIP $model frac=$frac seed=$seed (snapshot exists)"; continue
      fi
      echo "=== RUN $model frac=$frac seed=$seed ==="
      uv run python main.py $COMMON --model "$model" --data_fraction "$frac" --seed "$seed" \
        || { echo "FAILED $model frac=$frac seed=$seed"; exit 1; }
      rm -rf "$dst"; mkdir -p "$dst"
      cp results/"$model"/training_results/*.pth "$dst"/ \
        || { echo "no checkpoints to snapshot for $model"; exit 1; }
      echo "snapshotted -> $dst"
    done
  done
done
echo "DONE. Snapshots under $SNAP/. Baseline needs no sweep (fixed theta, flat reference)."
