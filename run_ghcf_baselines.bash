#!/usr/bin/env bash

set -uo pipefail
cd "$(dirname "$0")"

SEED=23
EPOCHS=10   # best epochs on the large datasets were <= 10; patience stays at the tuned 5
FOLDS=(0 1 2 3 4)
DATASETS=(Rotten_Tomatoes_data)
MODELS=(ConvMF DAML DeepCoNN KANN LETTER TARMF)
declare -A SETTING=([Imdb_data]=imdb_loocv_data [Amazon_data]=amazon_loocv_data
                   [Rotten_Tomatoes_data]=rotten_tomatoes_loocv_data)
# Rotten_Tomatoes_data was not tuned: results/optuna/Rotten_Tomatoes_data/*_tpe_best.json
# are copies of the Amazon_data ones (closest in catalog size, review length and 1-5 scale)

LOG_DIR=logs/ghcf
mkdir -p "$LOG_DIR"
FAILED=()

# PMF sizes ConvMF (fc_dim) and TARMF (id_emb_size) need, read from their best.json
pmf_dims() {
  python - "$1" <<'EOF'
import json, sys
ds = sys.argv[1]
dims = {json.load(open(f"results/optuna/{ds}/ConvMF_tpe_best.json"))["best_params"]["fc_dim"],
        json.load(open(f"results/optuna/{ds}/TARMF_tpe_best.json"))["best_params"]["id_emb_size"]}
print(",".join(map(str, sorted(dims))))
EOF
}

for ds in "${DATASETS[@]}"; do
  sp=${SETTING[$ds]}
  dims=$(pmf_dims "$ds")

  for k in "${FOLDS[@]}"; do
    echo "=================== $ds fold_$k ==================="

    # 1) per-fold inputs that the LOOCV build doesn't produce (skipped if already there)
    python add_eduardo/regen_amazon_fold0_letter_kann.py --dataset_dir="$sp" --fold="$k" --skip_rating_matrix \
      || { FAILED+=("$ds fold$k KANN-files"); }

    for n in ${dims//,/ }; do
      suffix=$([ "$n" = 32 ] && echo "" || echo "_pmf$n")
      if [ ! -f "checkpoints/user_features_pmf_${ds}_loocv_fold${k}${suffix}.npy" ]; then
        python preprocess_data/build_pmf_features.py --datasets="$ds" --setting_path="$sp" --fold="$k" \
          --emb_opt="loocv_fold$k" --n_factors_list="$n" || FAILED+=("$ds fold$k PMF$n")
      fi
    done

    # 2) train + evaluate each baseline with its best hyperparameters
    for m in "${MODELS[@]}"; do
      out_dir="results/ghcf/$ds/fold_$k"
      if [ -f "$out_dir/${m}_seed${SEED}.csv" ]; then
        echo "[skip] $ds fold_$k $m (already done)"
        continue
      fi
      python finalize_best_hyperparams.py --model="$m" --dataset="$ds" --seeds="[$SEED]" \
        --num_epochs="$EPOCHS" \
        --patience=3 \
        --emb_opt="loocv_fold$k" \
        --setting_path="$sp" --fold_idx="$k" \
        --results_dir="$out_dir" --checkpoint_dir="checkpoints/ghcf/$ds/fold_$k" \
        2>&1 | tee "$LOG_DIR/${ds}_fold${k}_${m}.log" \
        || FAILED+=("$ds fold$k $m")
    done
  done
done

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All runs finished."
else
  echo "Failed (see $LOG_DIR):"; printf '  %s\n' "${FAILED[@]}"
fi
