# -*- coding: utf-8 -*-
"""
Optuna-driven hyperparameter grid search for iRev models.

Search spaces are transcribed from update_baselines.md, with hyperparameter
names translated to the config attributes main.py's `opt.parse()` actually
recognizes (e.g. "learning_rate" -> "lr"). A few knobs named in that file
are not read by any model or by the training loop in main.py, which always
optimizes with `optim.Adam(lr=opt.lr, weight_decay=opt.weight_decay)`:
lambda_u/lambda_v (ConvMF, TARMF) and beta1/beta2/epsilon/optimizer (KANN).
Those have been dropped rather than added as inert config fields.

Each trial shells out to `python main.py train --tuning=True ...` so every
run gets a fresh CUDA/Python process (important with hundreds of trials in a
single grid). In tuning mode main.py trains on the fixed split (fold_0),
selects the epoch by validation MSE and prints a `GRID_SEARCH_METRIC=<value>`
line with the seed-averaged best validation MSE, which this script parses back
out as the value Optuna minimizes. BPR models (LightGCN) are selected by
validation nDCG@10 instead, so their metric is -nDCG@10 (still minimized).
Tuning mode skips checkpoints, results/new_protocol and the test split (and the
full-catalog ranking, except the validation ranking BPR models select on), so
trials never overwrite reported runs.

The best params (plus the epochs/patience/seeds they were tuned with) are saved
to <results_dir>/<dataset>/<model>_<sampler>_best.json, which
finalize_best_hyperparams.py reads.

Usage:
    python grid_search.py --model=LETTER --dataset=All_Beauty_data --sampler=tpe --max_trials=30
    python grid_search.py --model=CARL --dataset=All_Beauty_data --num_epochs=30 --patience=5 --seeds=[23]
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

import optuna
from tqdm import tqdm

PARAM_GRIDS = {
    "CARL": {
        "id_emb_size": [15, 50, 150, 200, 300],
        "filters_num": [50],
        "kernel_size": [3],
        "word_dim": [300],
        "weight_decay": [0.001, 0.005, 0.01, 0.05, 0.1],  # regularization_lambda
        "lr": [0.001],  # learning_rate
        "batch_size": [100, 200, 256],
    },
    "ConvMF": {
        "word_dim": [300],  # see word_dim note below
        "filters_num": [50, 100, 200],
        "kernel_size": [3, 4, 5],
        "fc_dim": [50, 100],
        "drop_out": [0.2, 0.5],
    },
    "DAML": {
        "weight_decay": [0.001, 0.01, 1, 10, 100],  # regularization
        "lr": [0.00001, 0.00005],  # learning_rate
        "id_emb_size": [8, 16, 32],
        "kernel_size": [3],
        "filters_num": [100],
        "word_dim": [300],  # see word_dim note below
    },
    "DeepCoNN": {
        "fc_dim": [25, 50, 100],
        "filters_num": [50, 100, 200],
        "kernel_size": [3],
        "word_dim": [300],  # see word_dim note below
        "drop_out": [0.2, 0.5],
        "weight_decay": [0.001, 0.002, 0.01],  # regularization_lambda
        "batch_size": [100],
    },
    # Paper: lr in {0.0005, 0.001, 0.005}, L2 in {0.001, 0.01, 0.05, 0.1}, d in {4..128}
    "APH": {
        "aph_dim": [8, 16, 32, 64],
        "lr": [0.0005, 0.001, 0.005],
        "weight_decay": [0.001, 0.01, 0.05, 0.1],
        "batch_size": [128, 256],
    },
    "KANN": {
        "kann_d_model": [50],  # see kann_d_model note below
        "kann_dff": [512, 1024, 2048],
        "kann_num_heads": [2, 4, 8],
        "kann_num_layers": [1, 2, 3],
        "drop_out": [0.1, 0.2, 0.5],
        "batch_size": [64, 128, 256],
    },
    "LETTER": {
        "letter_hidden_dim": [32, 64, 128],
        "drop_out": [0.3],
        "letter_edge_ratio": [1, 5, 10, 20],
        "lr": [0.0005, 0.00001, 0.001],  # learning_rate
        "batch_size": [128, 256],
    },
    # Paper: d=64, lr=0.001, batch 2048; searched K in {1..4} and L2 decay in {1e-5..1e-2}
    # Released code: lr 0.005, regularizer 8e-5 (never applied), ssl weight 0.05; batch as LightGCN
    "MAGCL": {
        "lr": [0.001, 0.005],
        "magcl_decay": [0, 0.00008, 0.001],
        "magcl_ssl_weight": [0.01, 0.05, 0.1],
        "batch_size": [2048],
    },
    # Released configs (amazon/yelp/steam): kd_weight 1e-3..1e-2, kd_temperature 0.1..0.2,
    # reg_weight 1e-8..1e-7; lr 1e-3, batch 4096
    "RLMRec": {
        "rlmrec_kd_weight": [0.001, 0.01, 0.1],
        "rlmrec_kd_temperature": [0.1, 0.2, 0.5],
        "rlmrec_reg_weight": [1e-8, 1e-7, 1e-6],
        "lr": [0.001],
        "batch_size": [4096],
    },
    "LightGCN": {
        "lgcn_layers": [1, 2, 3, 4],
        "lgcn_decay": [0.00001, 0.0001, 0.001, 0.01],
        "lr": [0.0005, 0.001, 0.005],
        "batch_size": [2048],
    },
    "TARMF": {
        "word_dim": [300],  # see word_dim note below
        "id_emb_size": [64, 128, 256],
    },
}
# NOTE on kann_d_model: models/kann.py loads pretrained kann_re_emb_path /
# kann_ori_emb_path .npy files whose embedding width is fixed at generation
# time (see add_eduardo/generate_dataset_kann.py's D_MODEL) -- it does not
# project them to opt.kann_d_model, it just nn.Embedding.from_pretrained()s
# them straight in and derives attn_dim = 4 * opt.kann_d_model from the
# *searched* value. Any opt.kann_d_model other than the dim the embeddings
# were generated at crashes with a shape mismatch in KANN.__init__'s
# LayerNorm, same failure mode as word_dim above. Pinned to 50 (config.py's
# DefaultConfig.kann_d_model default, and what every precomputed
# kann_entity_*_emb.npy in this repo is generated at) rather than searched.

# NOTE on word_dim: update_baselines.md's grids proposed searching word_dim
# (e.g. 100/200/512) for ConvMF/DAML/DeepCoNN/TARMF, but every model that
# uses word embeddings here hard-copies the pretrained w2v.npy straight into
# its embedding layer weights (e.g. models/daml.py's reset_para():
# self.user_word_embs.weight.data.copy_(w2v.cuda())). w2v.npy is generated
# from GoogleNews-vectors-negative300.bin, which is fixed at 300 dimensions
# -- any other word_dim crashes with a tensor size mismatch at model
# construction. word_dim is pinned to 300 (matching CARL's grid, which
# already had it right) rather than searched.

# Fixed (non-searched) architectural args each model requires to wire up
# correctly through framework/models.py -- these aren't hyperparameters,
# they're structural constants (e.g. num_fea = how many feature blocks the
# model's Net concatenates, which framework/models.py's generic feature_dim
# formula depends on). Values are transcribed from the working invocations
# in train.sh/test.sh. Without these, training crashes with a shape
# mismatch in the prediction head (feature_dim computed from opt.num_fea
# doesn't match what the model's forward() actually concatenates).
FIXED_ARGS = {
    "CARL": {"num_fea": 3},
    "ConvMF": {"num_fea": 1},
    "DAML": {"num_fea": 2},
    "DeepCoNN": {"num_fea": 1},
    "TARMF": {"num_fea": 2},
    # KANN/LETTER set self.num_fea = 1 on their Net internally and are meant
    # to bypass framework/models.py's generic fusion/prediction path via
    # opt.direct_output -- but they never set that flag themselves, so it
    # must be passed explicitly or the same feature_dim mismatch happens.
    "APH": {"direct_output": True},
    "KANN": {"direct_output": True},
    "LETTER": {"direct_output": True},
    "LightGCN": {"direct_output": True},
    "MAGCL": {"direct_output": True},
    "RLMRec": {"direct_output": True},
}

# Params whose value must be COPIED from another sampled param, because the
# model's actual concatenated feature width only matches framework/models.py's
# generic id_emb_size*num_fea*2 formula when the two are equal. E.g.
# DeepCoNN's Net concatenates fc_dim-wide user/item vectors, not
# id_emb_size-wide ones -- so id_emb_size must track whatever fc_dim was
# sampled for that trial, not be an independent free dimension.
DERIVED_ARGS = {
    "DeepCoNN": {"id_emb_size": "fc_dim"},
    "ConvMF": {"id_emb_size": "fc_dim"},
}

# model -> name of the searched param whose value must match the n_factors
# used to build that trial's PMF checkpoint (see
# preprocess_data/build_pmf_features.py). ConvMF concatenates its PMF user
# feature with an fc_dim-wide item feature; TARMF concatenates its PMF
# feature with an id_emb_size-wide projected feature -- in both cases
# framework/models.py's generic formula only comes out right if the PMF
# feature and the paired learned feature are the same width. A PMF
# checkpoint must exist for every value in that param's search grid (see
# preprocess_data/build_pmf_features.py --n_factors_list).
PMF_DIM_PARAM = {
    "ConvMF": "fc_dim",
    "TARMF": "id_emb_size",
}


def emb_opt_for_trial(model, params, base_emb_opt):
    dim_param = PMF_DIM_PARAM.get(model)
    if dim_param is None:
        return base_emb_opt
    n_factors = params[dim_param]
    return base_emb_opt if n_factors == 32 else f"{base_emb_opt}_pmf{n_factors}"


METRIC_RE = re.compile(r"GRID_SEARCH_METRIC=([\-0-9.eE]+)")


def grid_size(space):
    size = 1
    for values in space.values():
        size *= len(values)
    return size


# Files whose contents define what a trial trains on. Hashed into the study name so a
# rebuilt dataset (e.g. raw ratings instead of z-scores, or real word2vec instead of
# random vectors) starts a fresh study instead of resuming trials from the old data.
DATA_FINGERPRINT_FILES = ("Train.npy", "Train_Score.npy", "w2v.npy")
# Model-specific inputs, built by their own scripts: regenerating them (e.g. LETTER's
# embeddings restricted to train reviews) must also start a fresh study.
MODEL_FINGERPRINT_FILES = {
    "LETTER": ("Rating_Matrix_sparse.npz",) + tuple(
        f"letter_{side}_review_emb{suffix}.npy" for side in ("user", "item") for suffix in ("", "_pos", "_neg")),
    "KANN": ("kann_entity_re_emb.npy", "kann_entity_ori_emb.npy",
             "kann_user_entities.npy", "kann_item_entities.npy"),
    "APH": ("aph_user_edges.npy", "aph_item_edges.npy", "aph_stats.json"),
    "MAGCL": ("magcl_user_graph.npz", "magcl_item_graph.npz"),
    "RLMRec": ("rlmrec_user_prf_emb.npy", "rlmrec_item_prf_emb.npy"),
}


# Training-behaviour changes inside a model's code, which neither the grid nor the data
# fingerprint sees. Folded into the study name so trials trained before the change are not
# resumed. Only models listed here get the key, so other studies keep their names.
TRAINING_VARIANTS = {
    # models/aph.py: train pairs drop their own review's hyperedges; biases exempt from weight_decay
    "APH": {"aph_exclude_target": True, "bias_no_decay": True},
}


def data_fingerprint(dataset, model):
    import config
    train_dir = os.path.join("dataset", getattr(config, f"{dataset}_Config").setting_path, "fold_0", "train")
    h = hashlib.sha1()
    for name in DATA_FINGERPRINT_FILES + MODEL_FINGERPRINT_FILES.get(model, ()):
        path = os.path.join(train_dir, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:8]


def search_space_signature(search_space, settings):
    # Optuna pins each param's CategoricalDistribution (choices) to whatever
    # a study's first trial recorded, forever -- reusing a study_name after
    # PARAM_GRIDS changes (e.g. word_dim pinned from a wider grid down to
    # [300], see the word_dim note above) makes every later trial crash with
    # "CategoricalDistribution does not support dynamic value space.", even
    # though the stale trials it collides with never completed. Folding a
    # hash of the current grid into study_name means an edited grid always
    # starts a fresh study instead of colliding with incompatible old state;
    # the abandoned rows under the old name just sit unused in the same db.
    # The training settings (epochs, patience, seeds) are hashed too: trials
    # run under different budgets are not comparable and must not share a study.
    payload = json.dumps({"space": {k: list(v) for k, v in search_space.items()}, "settings": settings},
                         sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def run_trial(args, params):
    cmd = [
        args.python_bin, "main.py", "train",
        f"--model={args.model}",
        f"--dataset={args.dataset}",
        f"--num_epochs={args.num_epochs}",
        f"--patience={args.patience}",
        f"--seeds={args.seeds}",
        f"--emb_opt={emb_opt_for_trial(args.model, params, args.emb_opt)}",
        "--tuning=True",
    ]
    for k, v in FIXED_ARGS.get(args.model, {}).items():
        cmd.append(f"--{k}={v}")
    for k, src in DERIVED_ARGS.get(args.model, {}).items():
        cmd.append(f"--{k}={params[src]}")
    for k, v in params.items():
        cmd.append(f"--{k}={v}")
    cmd.extend(args.extra_args)

    env = os.environ.copy()
    if args.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    tqdm.write(f"[grid_search] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

    match = None
    for line in reversed(proc.stdout.splitlines()):
        match = METRIC_RE.search(line)
        if match:
            break

    if match is None:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-4000:])
        raise RuntimeError(
            f"Could not parse GRID_SEARCH_METRIC from trial output "
            f"(model={args.model}, dataset={args.dataset}, params={params})"
        )

    return float(match.group(1))


def make_objective(args, search_space):
    def objective(trial):
        params = {name: trial.suggest_categorical(name, values) for name, values in search_space.items()}
        return run_trial(args, params)

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter grid search for iRev models")
    parser.add_argument("--model", required=True, choices=sorted(PARAM_GRIDS))
    parser.add_argument("--dataset", required=True,
                        help="e.g. All_Beauty_data, Amazon_data, Imdb_data, Musical_Instruments_data, TripAdvisor_data")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5, help="épocas sem melhora no MSE da validação")
    parser.add_argument("--seeds", default="[23]", help="Python list literal forwarded to main.py train")
    parser.add_argument("--emb_opt", default="default")
    parser.add_argument("--max_trials", type=int, default=None,
                        help="cap on number of trials; default runs the full grid exhaustively")
    parser.add_argument("--sampler", choices=["grid", "tpe", "random"], default="grid",
                        help="'grid' (default) exhaustively visits every combo (respects --max_trials as a random "
                             "subset of the grid). 'tpe' and 'random' sample from the same search space instead of "
                             "enumerating it -- use with --max_trials for a fixed, adaptively-chosen trial budget.")
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="trials run concurrently (each its own main.py process on the same GPU). Small "
                             "models leave the GPU mostly idle, so 3-4 parallel trials give ~3x throughput.")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--study_dir", default="optuna_studies")
    parser.add_argument("--results_dir", default="results/optuna")
    parser.add_argument("--extra_args", nargs="*", default=[],
                        help="extra --key=value args forwarded verbatim to main.py train")
    args = parser.parse_args()

    search_space = PARAM_GRIDS[args.model]
    full_size = grid_size(search_space)

    if args.sampler == "grid":
        sampler = optuna.samplers.GridSampler(search_space)
        n_trials = args.max_trials or full_size
    else:
        if args.max_trials is None:
            parser.error(f"--sampler={args.sampler} requires --max_trials (there's no grid to exhaust)")
        # constant_liar: with --n_jobs > 1, keeps TPE from proposing the same params to
        # trials that start while others are still running (no effect when sequential)
        sampler = (optuna.samplers.TPESampler(seed=42, constant_liar=True) if args.sampler == "tpe"
                   else optuna.samplers.RandomSampler(seed=42))
        n_trials = args.max_trials

    study_dir = os.path.join(args.study_dir, args.dataset)
    os.makedirs(study_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(study_dir, args.model)}.db"
    settings = {"num_epochs": args.num_epochs, "patience": args.patience, "seeds": args.seeds,
                "data": data_fingerprint(args.dataset, args.model)}
    if not FIXED_ARGS.get(args.model, {}).get("direct_output"):
        # models scored through framework/prediction.py: a changed head (e.g. LFM's global
        # bias at the train mean) must not resume trials trained with the old one
        with open("framework/prediction.py", "rb") as f:
            settings["head"] = hashlib.sha1(f.read()).hexdigest()[:8]
    settings.update(TRAINING_VARIANTS.get(args.model, {}))
    study_name = f"{args.dataset}_{args.model}_{args.sampler}_{search_space_signature(search_space, settings)}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )

    completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = max(0, n_trials - completed)

    print(f"[grid_search] model={args.model} dataset={args.dataset} sampler={args.sampler} "
          f"grid_size={full_size} target_trials={n_trials} already_completed={completed} "
          f"remaining={remaining} storage={storage}")

    if remaining > 0:
        study.optimize(make_objective(args, search_space), n_trials=remaining, n_jobs=args.n_jobs,
                       show_progress_bar=True)
    else:
        print(f"[grid_search] target already reached, nothing to run")

    results_dir = os.path.join(args.results_dir, args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    study.trials_dataframe().to_csv(os.path.join(results_dir, f"{args.model}_{args.sampler}_trials.csv"), index=False)
    with open(os.path.join(results_dir, f"{args.model}_{args.sampler}_best.json"), "w") as f:
        json.dump({"study_name": study_name, "storage": storage, "best_value": study.best_value,
                   "best_params": study.best_params, "emb_opt": args.emb_opt, **settings}, f, indent=2)

    print(f"[grid_search] BEST for {args.model}/{args.dataset}: "
          f"value={study.best_value:.6f} params={study.best_params}")


if __name__ == "__main__":
    main()
