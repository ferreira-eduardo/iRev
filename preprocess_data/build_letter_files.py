"""Regenerates only LETTER's per-fold files (Rating_Matrix_sparse.npz +
letter_*_review_emb*.npy) without rerunning all of complement_folds.py.

Usage (from the repo root):
    .venv/bin/python preprocess_data/build_letter_files.py musical_instruments [beauty ...]

The fold split is recomputed exactly as complement_folds.py does it, and
checked against the fold's existing Train.npy so the LETTER files can never
silently disagree with the ids the rest of the pipeline was built on.
"""
import os
import sys

import numpy as np

from complement_folds import (REPO_ROOT, get_count, get_loocv_fold_normalized,
                              load_raw_dataset, remap_ids, save_letter_files)

if __name__ == "__main__":
    datasets = sys.argv[1:]
    if not datasets:
        sys.exit(__doc__)

    for dataset_name in datasets:
        print(f"=== {dataset_name} ===")
        df_full = load_raw_dataset(dataset_name)
        emb_dim = df_full['review_emb'].iloc[0].shape[0]

        for fold_idx in range(5):
            train_path = os.path.join(REPO_ROOT, "dataset", f"{dataset_name}_data",
                                      f"fold_{fold_idx}", "train")
            data_train, data_val, _ = get_loocv_fold_normalized(df_full, k_fold_index=fold_idx)
            u_map = {uid: i for i, uid in enumerate(get_count(data_train, 'user_id'))}
            i_map = {iid: i for i, iid in enumerate(get_count(data_train, 'item_id'))}
            data_train = remap_ids(data_train, u_map, i_map)
            data_val = remap_ids(data_val, u_map, i_map)

            expected = np.load(f"{train_path}/Train.npy")
            if not np.array_equal(expected, data_train[['user_id', 'item_id']].values):
                sys.exit(f"fold {fold_idx}: recomputed split does not match {train_path}/Train.npy; "
                         f"rerun complement_folds.py for {dataset_name} instead.")

            save_letter_files(train_path, data_train, data_val, len(u_map) + 2, len(i_map) + 2, emb_dim)
            print(f"fold {fold_idx}: wrote LETTER files to {train_path}")
