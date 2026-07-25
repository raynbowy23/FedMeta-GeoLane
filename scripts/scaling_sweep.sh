#!/usr/bin/env bash
# FLEET-SCALING SWEEP: unseen performance vs number of training cameras.
# Hypothesis: FedMeta's unseen advantage GROWS with fleet size (pooling
# aggregates scene->theta evidence across sites) while Meta's donor transfer
# scales only through better nearest-donor availability.
#
# n_train in {1, 3, 5}; 3 fixed camera draws per n (subset choice dominates at
# small n, so draws provide the error bars); meta + federated(perfedavg);
# seed 42. n=6 comes free from results/_final_batch (all 3 seeds). 18 runs,
# smaller fleets train faster. Resumable. Rei runs this (paper numbers).
#
#   bash scripts/scaling_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SNAP=results/_scaling
COMMON="--T 60 --is_save --use_historical_data --skip_continuous_learning --seed 42"
STALE=results/_pre_scaling_checkpoints

# Fixed draws (diverse mixes of busy/sparse and corridor position; n=5 = leave-one-out)
declare -A DRAWS=(
  [n1_d0]="US12_Monona"
  [n1_d1]="US12_Yahara"
  [n1_d2]="US12_Mineral"
  [n3_d0]="US12_Todd,US12_Yahara,US12_Mineral"
  [n3_d1]="US12_Monona,US12_CountyAB,US12_University"
  [n3_d2]="US12_Yahara,US12_Mineral,US12_University"
  [n5_d0]="US12_Monona,US12_Yahara,US12_CountyAB,US12_Mineral,US12_University"
  [n5_d1]="US12_Todd,US12_Yahara,US12_CountyAB,US12_Mineral,US12_University"
  [n5_d2]="US12_Todd,US12_Monona,US12_Yahara,US12_CountyAB,US12_University"
)

mkdir -p "$SNAP" "$STALE"
for model in federated meta; do
  if ls results/$model/training_results/*.pth >/dev/null 2>&1; then
    mv results/$model/training_results/*.pth "$STALE"/ 2>/dev/null || true
  fi
done

for key in n1_d0 n1_d1 n1_d2 n3_d0 n3_d1 n3_d2 n5_d0 n5_d1 n5_d2; do
  cams="${DRAWS[$key]}"
  for model in meta federated; do
    dst="$SNAP/${model}_${key}"
    if [ -f "$dst/.done" ]; then echo "SKIP $model $key"; continue; fi
    echo "=== RUN $model $key [$cams] ($(date +%H:%M)) ==="
    uv run python main.py $COMMON --model "$model" --seen_clients "$cams" \
      || { echo "FAILED $model $key"; exit 1; }
    rm -rf "$dst"; mkdir -p "$dst"
    cp results/"$model"/training_results/*.pth "$dst"/ 2>/dev/null || true
    echo "$cams" > "$dst/cameras.txt"
    touch "$dst/.done"
    echo "snapshotted -> $dst"
  done
done
echo "DONE ($(date +%H:%M)). n=6 point comes from results/_final_batch. Analyze with scripts/scaling_analysis.py"
