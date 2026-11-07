"""Builds iRev fold files (folds 0-4) from the ghc2f LOOCV splits, for the
checkpoints in checkpoints/loocv/.

    cd preprocess_data && ../.venv/bin/python build_loocv_folds.py amazon imdb rotten_tomatoes

Rows come from ghc2f/data/<name>.parquet and are labelled by
ghc2f/splits/<name>_fold<k>.csv (row_id == parquet row position). Everything
else (id maps, vocabulary, w2v, z-score targets, padding) is
complement_folds.write_split, the code that trained the loocv checkpoints.
Output: dataset/<name>_loocv_data/{common,fold_0..fold_4}.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

from complement_folds import REPO_ROOT, load_word2vec, write_split
from add_eduardo.leave_one_out_cv import normalize_by_train

GHC2F = os.environ.get("GHC2F_DIR", os.path.expanduser("~/PycharmProjects/ghc2f"))
N_FOLDS = 5
TEXT_COL = {"amazon": "text", "imdb": "text", "rotten_tomatoes": "review"}
LETTER_EMB_DIM = 768  # no review embeddings for these datasets: LETTER files are zeros


def load(name):
    raw = pd.read_parquet(f"{GHC2F}/data/{name}.parquet",
                          columns=["userId", "itemId", "rating", "timestamp", TEXT_COL[name]])
    df = pd.DataFrame({
        "user_id": raw["userId"].values,
        "item_id": raw["itemId"].values,
        "ratings": raw["rating"].astype(float).values,
        "reviews": raw[TEXT_COL[name]].fillna("").astype(str).values,
        "timestamp": raw["timestamp"].values,
    })
    zero = np.zeros(LETTER_EMB_DIM, dtype=np.float32)
    df["review_emb"] = [zero] * len(df)  # one shared array, no per-row copies
    return df


def main(names):
    pre_word2v = load_word2vec()
    for name in names:
        t0 = time.time()
        out = os.path.join(REPO_ROOT, "dataset", f"{name}_loocv_data")
        common = os.path.join(out, "common")
        os.makedirs(common, exist_ok=True)
        df = load(name)
        # same ordering as add_eduardo.leave_one_out_cv.get_loocv_fold_normalized
        srt = df.sort_values(["user_id", "timestamp"])
        global_stats = {}  # vocabulary/w2v fixed on fold 0, shared by all folds
        for k in range(N_FOLDS):
            split = pd.read_csv(f"{GHC2F}/splits/{name}_fold{k}.csv")
            assert len(split) == len(df) and (split["row_id"].values == np.arange(len(df))).all()
            assert (split["userId"].values == df["user_id"].values).all()
            assert (split["itemId"].values == df["item_id"].values).all()
            lab = split["split"].values[srt.index]
            train, val, test = (srt[lab == s] for s in ("train", "val", "test"))
            train, val, test = normalize_by_train(train, val, test)
            print(f"[{name}] fold {k}: train {len(train):,} val {len(val):,} test {len(test):,}", flush=True)
            write_split(os.path.join(out, f"fold_{k}"), common, train, val, test, pre_word2v, global_stats)
        print(f"[{name}] done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["amazon", "imdb", "rotten_tomatoes"])
