#!/usr/bin/env bash
# Waits for the Musical Instruments build (already running) and runs
# hyper_param_grid_search_mi.bash right away, sharing the GPU with the IMDB grid search.
#     nohup bash run_mi_after_imdb.bash > logs/run_mi_after_imdb.log 2>&1 &
cd "$(dirname "$0")"
TRAIN=dataset/musical_instruments_data/fold_0/train

while pgrep -f "build_fixed_dataset.sh musical_instruments" > /dev/null; do sleep 60; done
if grep -qE "Traceback|Killed|MemoryError" logs/build_musical_instruments.log || [ ! -f $TRAIN/aph_stats.json ]; then
    echo "[$(date)] build FAILED, see logs/build_musical_instruments.log; not running the grid search"
    exit 1
fi
if grep -q "Word2Vec não carregado" logs/build_musical_instruments.log; then
    echo "[$(date)] build used RANDOM word2vec (GoogleNews .bin missing); not running the grid search"
    exit 1
fi
echo "[$(date)] build done, starting Musical Instruments (in parallel with IMDB)"
bash hyper_param_grid_search_mi.bash > logs/hyper_param_grid_search_mi.log 2>&1
echo "[$(date)] Musical Instruments finished (exit $?)"
