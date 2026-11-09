#!/usr/bin/env bash
# LightGCN on the LOOCV folds with the tuned hyperparameters
# (results/optuna/<dataset>/LightGCN_tpe_best.json). Results go next to the other
# GHCF baselines, so each fold's summary.csv includes LightGCN.

set -uo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1   # print() lines show up in the log as they happen

SEED=23
FOLDS=(0 1 2 3 4)
DATASETS=(Rotten_Tomatoes_data Amazon_data Imdb_data)
declare -A SETTING=([Rotten_Tomatoes_data]=rotten_tomatoes_loocv_data
                   [Amazon_data]=amazon_loocv_data [Imdb_data]=imdb_loocv_data)
# LightGCN was tuned only on Rotten_Tomatoes_data: results/optuna/{Amazon_data,Imdb_data}/
# LightGCN_tpe_best.json are copies of that one (see their "copied_from" field)
#
# num_epochs/patience: the tuned budget (1000 / 5) comes from the best.json. BPR models
# rank the validation set every 10 epochs and patience counts those evaluations, so
# training stops after 50 epochs without a better val nDCG@10.

LOG_DIR=logs/ghcf
mkdir -p "$LOG_DIR"
FAILED=()

for ds in "${DATASETS[@]}"; do
  sp=${SETTING[$ds]}
  for k in "${FOLDS[@]}"; do
    out_dir="results/ghcf/$ds/fold_$k"
    if [ -f "$out_dir/LightGCN_seed${SEED}.csv" ]; then
      echo "[skip] $ds fold_$k LightGCN (already done)"
      continue
    fi
    echo "=================== $ds fold_$k LightGCN ==================="
    python finalize_best_hyperparams.py --model=LightGCN --dataset="$ds" --seeds="[$SEED]" \
      --emb_opt="loocv_fold$k" \
      --setting_path="$sp" --fold_idx="$k" \
      --results_dir="$out_dir" --checkpoint_dir="checkpoints/ghcf/$ds/fold_$k" \
      2>&1 | tee "$LOG_DIR/${ds}_fold${k}_LightGCN.log" \
      || FAILED+=("$ds fold$k LightGCN")
  done
done

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All LightGCN runs finished."
else
  echo "Failed (see $LOG_DIR):"; printf '  %s\n' "${FAILED[@]}"
fi
