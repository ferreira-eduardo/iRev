# -*- coding: utf-8 -*-
"""
Retrains a model with the best hyperparameters found by grid_search.py and
evaluates it under the seed protocol (fixed split, full-catalog ranking).

The search itself (main.py train --tuning=True) only reports validation MSE;
this script runs the normal `main.py train`, which selects the epoch by
validation MSE and writes val/test reports to
results/new_protocol/<dataset>/<model>_seed<seed>.csv (+ summary.csv).

By default it reuses the num_epochs/patience the search was run with, so the
reported model gets the same training budget its hyperparameters were tuned for.

Usage:
    python finalize_best_hyperparams.py --model=LETTER --dataset=All_Beauty_data --sampler=tpe --seeds=[23]
"""
import argparse
import json
import os
import subprocess
import sys

from grid_search import PARAM_GRIDS, FIXED_ARGS, DERIVED_ARGS, emb_opt_for_trial


def load_best(results_dir, dataset, model, sampler):
    path = os.path.join(results_dir, dataset, f"{model}_{sampler}_best.json")
    if not os.path.exists(path):
        raise RuntimeError(f"{path} não existe: rode grid_search.py --model={model} "
                           f"--dataset={dataset} --sampler={sampler} primeiro.")
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Retrain+evaluate a model's best hyperparameters over seeds")
    parser.add_argument("--model", required=True, choices=sorted(PARAM_GRIDS))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sampler", default="tpe", choices=["grid", "tpe", "random"],
                         help="which grid_search.py study to pull best params from")
    parser.add_argument("--seeds", default="[23]", help="Python list literal of seeds for the reported runs")
    parser.add_argument("--num_epochs", type=int, default=None, help="default: the value used in the search")
    parser.add_argument("--patience", type=int, default=None, help="default: the value used in the search")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--search_results_dir", default="results/optuna")
    args = parser.parse_args()

    best = load_best(args.search_results_dir, args.dataset, args.model, args.sampler)
    best_params = best["best_params"]
    num_epochs = args.num_epochs or best["num_epochs"]
    patience = args.patience or best["patience"]
    print(f"[finalize] best params for {args.model}/{args.dataset} "
          f"(search val MSE={best['best_value']:.6f}, study {best['study_name']}): {best_params}")

    cmd = [args.python_bin, "main.py", "train", f"--model={args.model}", f"--dataset={args.dataset}",
           f"--seeds={args.seeds}", f"--num_epochs={num_epochs}", f"--patience={patience}",
           # emb_opt picks the PMF checkpoint (ConvMF/TARMF) and is part of the model
           # checkpoint filename, so it must match what the trial used
           f"--emb_opt={emb_opt_for_trial(args.model, best_params, best['emb_opt'])}"]
    cmd += [f"--{k}={v}" for k, v in FIXED_ARGS.get(args.model, {}).items()]
    cmd += [f"--{k}={best_params[src]}" for k, src in DERIVED_ARGS.get(args.model, {}).items()]
    cmd += [f"--{k}={v}" for k, v in best_params.items()]

    print(f"[finalize] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed (exit {proc.returncode}): {' '.join(cmd)}")
    print(f"[finalize] {args.model}/{args.dataset} DONE -- results under results/new_protocol/{args.dataset}/")


if __name__ == "__main__":
    main()
