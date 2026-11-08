#!/usr/bin/env bash

SEEDS="[23,43,2]"

#python grid_search.py --model=LETTER --dataset=All_Beauty_data --sampler=tpe --max_trials=30
#python grid_search.py --model=APH --dataset=All_Beauty_data --sampler=tpe --max_trials=30
#python grid_search.py --model=LightGCN --dataset=All_Beauty_data --sampler=tpe --max_trials=30 --num_epochs=1000 --patience=5

# IMDB (bash preprocess_data/build_fixed_dataset.sh imdb)
#python grid_search.py --model=LETTER --dataset=Imdb_data --sampler=tpe --max_trials=30
#python grid_search.py --model=APH --dataset=Imdb_data --sampler=tpe --max_trials=30
#python grid_search.py --model=LightGCN --dataset=Imdb_data --sampler=tpe --max_trials=30 --num_epochs=1000 --patience=5

#python finalize_best_hyperparams.py --model=LETTER --dataset=Imdb_data --sampler=tpe --seeds=$SEEDS
python finalize_best_hyperparams.py --model=APH --dataset=Imdb_data --sampler=tpe --seeds=$SEEDS
python finalize_best_hyperparams.py --model=LightGCN --dataset=Imdb_data --sampler=tpe --seeds=$SEEDS


