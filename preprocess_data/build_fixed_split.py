"""Builds an iRev dataset from a fixed train/val/test split (no LOOCV folds).

The split is read as-is from dataset/<name>/splits or dataset/<name> (splits_location.py;
cut from ~/dataset/<name>.parquet by build_splits.py), so every baseline is trained and
evaluated on exactly the same interactions. The output goes to dataset/<name>_data/fold_0,
the layout config.py already reads.

The *_Score.npy targets are the raw ratings: with most users having a single train
interaction, a per-user z-score degenerates (target 0 for those users). Relevance for
the ranking metrics is defined at evaluation time by new_evaluation.user_thresholds.

    python preprocess_data/build_fixed_split.py all_beauty    # IREV_SPLITS_DIR=... to override
"""
import os
import sys
import time

import numpy as np
import pandas as pd

from complement_folds import REPO_ROOT, load_word2vec, write_split
from splits_location import splits_dir as default_splits_dir

REVIEW_EMB_DIM = 768


def read_split(splits_dir, name):
    df = pd.read_parquet(os.path.join(splits_dir, f"{name}.parquet"))
    # user_id é o hash string da Amazon; a chave numérica é userId
    df = df.drop(columns="user_id", errors="ignore").rename(columns={"userId": "user_id", "itemId": "item_id",
                            "rating": "ratings", "review": "reviews"})
    df = df[["user_id", "item_id", "ratings", "reviews", "timestamp"]].copy()
    df["ratings"] = df["ratings"].astype(float)
    df["reviews"] = df["reviews"].fillna("").astype(str)
    # sem embeddings de review alinhados ao split: LETTER recebe zeros
    df["review_emb"] = list(np.zeros((len(df), REVIEW_EMB_DIM), dtype=np.float32))
    return df


if __name__ == "__main__":
    start_time = time.time()
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "all_beauty"
    splits_dir = default_splits_dir(dataset_name)

    data_root = os.path.join(REPO_ROOT, "dataset", f"{dataset_name}_data")
    common_path = os.path.join(data_root, "common")
    os.makedirs(common_path, exist_ok=True)

    train, val, test = (read_split(splits_dir, n) for n in ("train", "val", "test"))
    print(f"{dataset_name}: train={len(train)} val={len(val)} test={len(test)} (de {splits_dir})")

    write_split(os.path.join(data_root, "fold_0"), common_path, train, val, test,
                load_word2vec(), global_stats={})
    print(f"\nFinalizado! Tempo total: {time.time() - start_time:.2f}s")
