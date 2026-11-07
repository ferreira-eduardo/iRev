#!/usr/bin/env bash

SEEDS="[23,43,2]"

python grid_search.py --model=RLMRec --dataset=All_Beauty_data --sampler=tpe --max_trials=30 --num_epochs=1000 --patience=5
python grid_search.py --model=RLMRec --dataset=Imdb_data --sampler=tpe --max_trials=30 --num_epochs=1000 --patience=5
python grid_search.py --model=RLMRec --dataset=TripAdvisor_data --sampler=tpe --max_trials=30 --num_epochs=1000 --patience=5
python grid_search.py --model=RLMRec --dataset=Musical_Instruments_data --sampler=tpe --max_trials=30 --num_epochs=1000 --patience=5

python finalize_best_hyperparams.py --model=RLMRec --dataset=All_Beauty_data --sampler=tpe --seeds=$SEEDS
python finalize_best_hyperparams.py --model=RLMRec --dataset=Imdb_data --sampler=tpe --seeds=$SEEDS
python finalize_best_hyperparams.py --model=RLMRec --dataset=TripAdvisor_data --sampler=tpe --seeds=$SEEDS
python finalize_best_hyperparams.py --model=RLMRec --dataset=Musical_Instruments_data --sampler=tpe --seeds=$SEEDS
