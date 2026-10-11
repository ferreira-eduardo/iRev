# -*- coding: utf-8 -*-
"""
Builds the PMF-style user/item latent-feature checkpoints that ConvMF and
TARMF (models/convmf.py, models/tarmf.py) require at construction time:
checkpoints/user_features_pmf_<dataset>_<emb_opt>.npy
checkpoints/item_features_pmf_<dataset>_<emb_opt>.npy

Uses SVD (scikit-surprise) fit on fold_0's train interactions -- the same
approach as add_eduardo/user_item_feat_reviews.py, but reading directly from
this project's own preprocessed Train.npy/Train_Score.npy instead of a raw
csv, so it works for any dataset already run through complement_folds.py.

Only fold_0 is used: convmf.py/tarmf.py load a single file per dataset (not
per fold), so one shared PMF fit is reused across all 5 folds -- consistent
with how w2v.npy/vocab are already shared via dataset/<name>_data/common/.

Usage:
    python build_pmf_features.py --datasets=Amazon_data,Imdb_data,Beauty_data
"""
import argparse
import os

import numpy as np
import pandas as pd
from surprise import Reader, Dataset, SVD

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# *_Config.setting_path values (config/config.py) -- dataset dir names differ
# from the *_Config class names used on the CLI / in checkpoint filenames.
SETTING_PATHS = {
    "All_Beauty_data": "all_beauty_data",
    "Amazon_data": "amazon_data",
    "Imdb_data": "imdb_data",
    "Beauty_data": "beauty_data",
    "Musical_Instruments_data": "musical_instruments_data",
    "TripAdvisor_data": "tripadvisor_data",
}


def build_pmf_features(dataset_config_name, emb_opt="default", n_factors=32,
                        n_epochs=30, lr_all=0.005, reg_all=0.02, seed=42, out_emb_opt=None):
    """out_emb_opt: filename suffix to save under, if different from emb_opt
    (used to keep several n_factors variants of the same dataset alongside
    each other -- see grid_search.py's PMF_DIM_PARAM)."""
    out_emb_opt = out_emb_opt or emb_opt
    setting_path = SETTING_PATHS[dataset_config_name]
    train_dir = os.path.join(REPO_ROOT, "dataset", setting_path, "fold_0", "train")

    x = np.load(os.path.join(train_dir, "Train.npy"))
    y = np.load(os.path.join(train_dir, "Train_Score.npy"))

    df = pd.DataFrame({"userId": x[:, 0], "itemId": x[:, 1], "rating": y})
    rating_scale = (float(y.min()), float(y.max()))

    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(df[["userId", "itemId", "rating"]], reader)
    trainset = data.build_full_trainset()

    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all,
               reg_all=reg_all, random_state=seed)
    algo.fit(trainset)

    # algo.pu/qi are indexed by surprise's *internal* ids, not our raw
    # userId/itemId -- remap back so raw_u_feat[i] matches user i.
    user_features = np.zeros((trainset.n_users, n_factors), dtype=np.float32)
    item_features = np.zeros((trainset.n_items, n_factors), dtype=np.float32)
    for inner_uid in trainset.all_users():
        user_features[int(trainset.to_raw_uid(inner_uid))] = algo.pu[inner_uid]
    for inner_iid in trainset.all_items():
        item_features[int(trainset.to_raw_iid(inner_iid))] = algo.qi[inner_iid]

    out_dir = os.path.join(REPO_ROOT, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    user_path = os.path.join(out_dir, f"user_features_pmf_{dataset_config_name}_{out_emb_opt}.npy")
    item_path = os.path.join(out_dir, f"item_features_pmf_{dataset_config_name}_{out_emb_opt}.npy")
    np.save(user_path, user_features)
    np.save(item_path, item_features)
    print(f"[build_pmf_features] {dataset_config_name}: saved {user_features.shape} -> {user_path}")
    print(f"[build_pmf_features] {dataset_config_name}: saved {item_features.shape} -> {item_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="Amazon_data,Imdb_data,Beauty_data")
    parser.add_argument("--emb_opt", default="default")
    parser.add_argument("--n_factors_list", default="32",
                         help="comma-separated n_factors values to build -- ConvMF/TARMF's "
                              "id_emb_size/fc_dim search grids need a matching PMF file per "
                              "distinct value (see grid_search.py's PMF_DIM_PARAM), saved as "
                              "emb_opt=<emb_opt>_pmf<n_factors>.")
    args = parser.parse_args()

    for name in args.datasets.split(","):
        for n_factors in [int(x) for x in args.n_factors_list.split(",")]:
            print(f"\n{'=' * 40}\n{name} (n_factors={n_factors})\n{'=' * 40}")
            out_emb_opt = args.emb_opt if n_factors == 32 else f"{args.emb_opt}_pmf{n_factors}"
            build_pmf_features(name, emb_opt=args.emb_opt, n_factors=n_factors, out_emb_opt=out_emb_opt)
