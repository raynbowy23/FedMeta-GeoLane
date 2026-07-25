#!/usr/bin/env bash
# FINAL TABLE-OF-RECORD BATCH: 3 seeds x {baseline, meta, federated(perfedavg),
# federated(fedavg ablation)} on the 6-seen split (uniform >5m exclusion rule).
# The fedavg ablation is REQUIRED: perfedavg has never been compared to fedavg
# post-lateral, and the paper may only claim the meta-init matters if it
# survives this comparison across seeds. Rei runs this (paper numbers).
# Resumable: skips runs whose snapshot already exists. ~12h for the full grid.
#
#   bash scripts/final_batch.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(42 43 44)
SNAP=results/_final_batch
COMMON="--T 60 --is_save --use_historical_data --skip_continuous_learning"
STALE=results/_pre_final_checkpoints

mkdir -p "$SNAP" "$STALE"
# park any live checkpoints once so the first run starts clean
for model in federated meta baseline; do
  if ls results/$model/training_results/*.pth >/dev/null 2>&1; then
    mv results/$model/training_results/*.pth "$STALE"/ 2>/dev/null || true
  fi
done

run_one () {  # run_one <model> <seed> <tag> [extra args]
  local model=$1 seed=$2 tag=$3; shift 3
  local dst="$SNAP/${tag}_s${seed}"
  if [ -f "$dst/.done" ]; then
    echo "SKIP $tag seed=$seed (done)"; return 0
  fi
  echo "=== RUN $tag seed=$seed ($(date +%H:%M)) ==="
  uv run python main.py $COMMON --model "$model" --seed "$seed" "$@" \
    || { echo "FAILED $tag seed=$seed"; exit 1; }
  rm -rf "$dst"; mkdir -p "$dst"
  # baseline has no model, hence no checkpoints — the marker alone records it
  # (its numbers live in MLflow; its fixed theta needs no snapshot)
  cp results/"$model"/training_results/*.pth "$dst"/ 2>/dev/null || true
  touch "$dst/.done"
  echo "snapshotted -> $dst"
}

for seed in "${SEEDS[@]}"; do
  run_one baseline  "$seed" baseline
  run_one meta      "$seed" meta
  run_one federated "$seed" fed_perfedavg
  run_one federated "$seed" fed_fedavg --fed_algo fedavg
done

echo "=== DONE ($(date +%H:%M)). Snapshots under $SNAP/. ==="
echo "Score each with strategy_annotation_eval.py --fed_ckpt/--meta_ckpt_dir per snapshot,"
echo "or ask Claude to aggregate (mean+/-std over seeds; claims only where they survive error bars)."
