#!/usr/bin/env bash
# MAGCL end to end, one dataset per call, from the fixed splits in ~/ghc2f/data/<dataset>/splits.
# Only splits/train.parquet is encoded: the review-similarity graphs never see val/test text.
# The data/encode/graphs steps skip themselves when their output exists, so it is safe to
# re-run after an interruption.
#
#   bash run_magcl.bash <dataset> [step]
#   dataset: all_beauty | imdb | musical_instruments | tripadvisor
#   step:    all (default) | data | encode | graphs | grid | finalize
#
#   data      fixed split -> dataset/<ds>_data (the arrays main.py reads)
#   encode    all-MiniLM-L6-v2 vectors of the train reviews -> embeddings_reviews/<ds>_train_minilm.npy
#   graphs    user-user / item-item cosine > 0.83 graphs -> magcl_{user,item}_graph.npz
#   grid      grid_search.py over lr, magcl_decay and magcl_ssl_weight
#   finalize  finalize_best_hyperparams.py: best params retrained with SEEDS, reported on test
#
# Long steps (grid, finalize) are best run with nohup:
#   nohup bash run_magcl.bash all_beauty > logs/magcl_all_beauty.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"

ds=${1:?dataset: all_beauty | imdb | musical_instruments | tripadvisor}
step=${2:-all}

PY=${PY:-$HOME/ghc2f/.venv/bin/python}   # torch, pandas, sentence-transformers, optuna (+ gensim, fire)
SEEDS=${SEEDS:-[23,43,2]}
export IREV_SPLITS_DIR=${GHC2F:-$HOME/ghc2f}/data/$ds/splits

declare -A CONFIG=([all_beauty]=All_Beauty_data [imdb]=Imdb_data
                   [musical_instruments]=Musical_Instruments_data [tripadvisor]=TripAdvisor_data)
path=dataset/${ds}_data/
mkdir -p logs

run() { [ "$step" = all ] || [ "$step" = "$1" ]; }

if run data && [ ! -f "$path/user_ids.npy" ]; then
    echo "== $ds: model data ($(date))"
    $PY preprocess_data/build_fixed_split.py "$ds"
fi

if run encode && [ ! -f "embeddings_reviews/${ds}_train_minilm.npy" ]; then
    echo "== $ds: review embeddings ($(date))"
    $PY preprocess_data/encode_magcl_reviews.py "$ds"
fi

if run graphs && [ ! -f "$path/train/magcl_item_graph.npz" ]; then
    echo "== $ds: similarity graphs ($(date))"
    $PY preprocess_data/build_magcl_fixed_split.py "$ds"
fi

if run grid; then
    echo "== $ds: grid search ($(date))"
    $PY grid_search.py --model=MAGCL --dataset="${CONFIG[$ds]}" --sampler=tpe \
        --max_trials=30 --num_epochs=1000 --patience=5
fi

if run finalize; then
    echo "== $ds: finalize best hyperparameters, seeds $SEEDS ($(date))"
    $PY finalize_best_hyperparams.py --model=MAGCL --dataset="${CONFIG[$ds]}" --sampler=tpe \
        --seeds="$SEEDS" --python_bin="$PY"
fi
echo "== $ds: done ($(date))"
