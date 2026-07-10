#!/usr/bin/env bash
# Full three-strategy retrain after the lateral-coordinate histogram fix
# (detection semantics changed; all prior checkpoints and tables are stale).
# Rei runs this. Error-gated: stops on the first failed run.
#
#   bash scripts/retrain_lateral.sh
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="--T 60 --is_save --use_historical_data --skip_continuous_learning"
STALE=results/_pre_lateral_checkpoints

# Move old-semantics checkpoints aside (idempotent; skips if already moved)
mkdir -p "$STALE"
for model in federated meta baseline; do
  if ls results/$model/training_results/*.pth >/dev/null 2>&1; then
    mv results/$model/training_results/*.pth "$STALE"/
    echo "moved stale $model checkpoints -> $STALE/"
  fi
done

for model in baseline meta federated; do
  echo "=== RETRAIN $model ($(date +%H:%M)) ==="
  uv run python main.py $COMMON --model "$model" \
    || { echo "FAILED: $model"; exit 1; }
done

echo "=== DONE ($(date +%H:%M)). Score with: ==="
echo "uv run python scripts/strategy_annotation_eval.py --taus 3 5 --tag post-lateral"
