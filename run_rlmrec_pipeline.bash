#!/usr/bin/env bash
# RLMRec end to end, one dataset per call, from the fixed splits in ~/ghc2f/data/<dataset>/splits.
# Only splits/train.parquet is used for the profiles; ~/ghc2f/data/<dataset>/profiles (ghc2f's own
# profiles) is never read.
# Each step skips itself when its output exists, so it is safe to re-run after an interruption.
#
#   bash run_rlmrec_pipeline.bash <dataset> [step]
#   dataset: all_beauty | imdb | musical_instruments | tripadvisor
#   step:    all (default) | data | profiles | embeddings | grid
#
#   data       fixed split -> dataset/<ds>_data (the arrays main.py reads)
#   profiles   Qwen3-14B user/item profiles from TRAIN reviews (whole GPU, resumable)
#   embeddings e5-base-v2 encoding of the profiles -> rlmrec_{user,item}_prf_emb.npy
#   grid       grid_search.py over RLMRec's alignment weight, temperature and l2
#
# Long steps (profiles, grid) are best run with nohup:
#   nohup bash run_rlmrec_pipeline.bash all_beauty profiles > logs/rlmrec_all_beauty_profiles.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"

ds=${1:?dataset: all_beauty | imdb | musical_instruments | tripadvisor}
step=${2:-all}

PY=${PY:-$HOME/ghc2f/.venv/bin/python}   # torch, pandas, sentence-transformers, optuna (+ gensim, fire)
VLLM_PY=.venv-vllm/bin/python            # vLLM 0.31, only for the profiles
export IREV_SPLITS_DIR=${GHC2F:-$HOME/ghc2f}/data/$ds/splits

declare -A CONFIG=([all_beauty]=All_Beauty_data [imdb]=Imdb_data
                   [musical_instruments]=Musical_Instruments_data [tripadvisor]=TripAdvisor_data)
path=dataset/${ds}_data/
mkdir -p logs

run() { [ "$step" = all ] || [ "$step" = "$1" ]; }

# No GoogleNews word2vec in language_models/: w2v.npy falls back to seeded random vectors,
# which RLMRec does not use.
if run data && [ ! -f "$path/user_ids.npy" ]; then
    echo "== $ds: model data ($(date))"
    $PY preprocess_data/build_fixed_split.py "$ds"
fi

if run profiles; then
    echo "== $ds: profiles ($(date))"
    PATH="$PWD/.venv-vllm/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0 \
    RLMREC_QUANT=${RLMREC_QUANT:-fp8} RLMREC_GPU_UTIL=${RLMREC_GPU_UTIL:-0.85} \
        $VLLM_PY preprocess_data/generate_rlmrec_profiles.py "$ds"
fi

if run embeddings && [ ! -f "$path/train/rlmrec_item_prf_emb.npy" ]; then
    echo "== $ds: embeddings ($(date))"
    $PY preprocess_data/build_rlmrec_fixed_split.py "$ds"
fi

if run grid; then
    echo "== $ds: grid search ($(date))"
    $PY grid_search.py --model=RLMRec --dataset="${CONFIG[$ds]}" --sampler=tpe \
        --max_trials=30 --num_epochs=1000 --patience=5
fi
echo "== $ds: done ($(date))"
