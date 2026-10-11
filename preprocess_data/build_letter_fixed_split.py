"""Rewrites only LETTER's files (Rating_Matrix_sparse.npz + letter_*_review_emb*.npy)
for a fixed-split dataset built by build_fixed_split.py, using real review
embeddings instead of the zeros build_fixed_split.py writes.

    ~/PycharmProjects/ghc2f/.venv/bin/python preprocess_data/encode_split_reviews.py all_beauty
    .venv/bin/python preprocess_data/build_letter_fixed_split.py all_beauty

Ids are remapped with fold_0's saved user_ids.npy/item_ids.npy and checked
against fold_0/train/Train.npy, so the files cannot disagree with the rest of fold_0.
"""
import os
import sys

import numpy as np

from build_fixed_split import read_split
from complement_folds import REPO_ROOT, remap_ids, save_letter_files
from splits_location import splits_dir as default_splits_dir

EMB_DIR = os.path.join(REPO_ROOT, "embeddings_reviews")

if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "all_beauty"
    splits_dir = default_splits_dir(dataset_name)
    fold_folder = os.path.join(REPO_ROOT, "dataset", f"{dataset_name}_data", "fold_0")
    train_path = os.path.join(fold_folder, "train")

    splits = {}
    for split in ("train", "val"):
        df = read_split(splits_dir, split)
        emb = np.load(os.path.join(EMB_DIR, f"{dataset_name}_{split}.npy"))
        if len(emb) != len(df):
            sys.exit(f"{split}: {len(emb)} embeddings para {len(df)} linhas; rode encode_split_reviews.py de novo.")
        df["review_emb"] = list(emb)
        splits[split] = df

    u_map = {uid: i for i, uid in enumerate(np.load(f"{fold_folder}/user_ids.npy"))}
    i_map = {iid: i for i, iid in enumerate(np.load(f"{fold_folder}/item_ids.npy"))}
    data_train = remap_ids(splits["train"], u_map, i_map)
    data_val = remap_ids(splits["val"], u_map, i_map)

    expected = np.load(f"{train_path}/Train.npy")
    if not np.array_equal(expected, data_train[["user_id", "item_id"]].values):
        sys.exit(f"split recomputado não bate com {train_path}/Train.npy; rode build_fixed_split.py de novo.")

    save_letter_files(train_path, data_train, data_val, len(u_map) + 2, len(i_map) + 2, emb.shape[1])
    print(f"LETTER files escritos em {train_path}")
