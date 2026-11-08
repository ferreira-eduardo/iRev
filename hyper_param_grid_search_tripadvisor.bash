#!/usr/bin/env bash
# TripAdvisor (bash preprocess_data/build_fixed_dataset.sh tripadvisor;
# RLMRec also needs: bash run_rlmrec_profiles.bash tripadvisor)

SEEDS="[23,43,2]"

python grid_search.py --model=LETTER --dataset=TripAdvisor_data --sampler=tpe --max_trials=30
python grid_search.py --model=APH --dataset=TripAdvisor_data --sampler=tpe --max_trials=30
python grid_search.py --model=LightGCN --dataset=TripAdvisor_data --sampler=tpe --max_trials=30 --num_epochs=1000 --patience=5
#python grid_search.py --model=MAGCL --dataset=TripAdvisor_data --sampler=tpe --max_trials=30 --num_epochs=1000 --patience=5
#python grid_search.py --model=RLMRec --dataset=TripAdvisor_data --sampler=tpe --max_trials=30 --num_epochs=1000 --patience=5

python finalize_best_hyperparams.py --model=LETTER --dataset=TripAdvisor_data --sampler=tpe --seeds=$SEEDS
python finalize_best_hyperparams.py --model=APH --dataset=TripAdvisor_data --sampler=tpe --seeds=$SEEDS
python finalize_best_hyperparams.py --model=LightGCN --dataset=TripAdvisor_data --sampler=tpe --seeds=$SEEDS
#python finalize_best_hyperparams.py --model=MAGCL --dataset=TripAdvisor_data --sampler=tpe --seeds=$SEEDS
#python finalize_best_hyperparams.py --model=RLMRec --dataset=TripAdvisor_data --sampler=tpe --seeds=$SEEDS
