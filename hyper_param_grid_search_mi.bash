#!/usr/bin/env bash
# Musical Instruments (bash preprocess_data/build_fixed_dataset.sh musical_instruments)

SEEDS="[23,43,2]"

python grid_search.py --model=LETTER --dataset=Musical_Instruments_data --sampler=tpe --max_trials=30
python grid_search.py --model=APH --dataset=Musical_Instruments_data --sampler=tpe --max_trials=30
python grid_search.py --model=LightGCN --dataset=Musical_Instruments_data --sampler=tpe --max_trials=30 --num_epochs=1000 --patience=5

python finalize_best_hyperparams.py --model=LETTER --dataset=Musical_Instruments_data --sampler=tpe --seeds=$SEEDS
python finalize_best_hyperparams.py --model=APH --dataset=Musical_Instruments_data --sampler=tpe --seeds=$SEEDS
python finalize_best_hyperparams.py --model=LightGCN --dataset=Musical_Instruments_data --sampler=tpe --seeds=$SEEDS
