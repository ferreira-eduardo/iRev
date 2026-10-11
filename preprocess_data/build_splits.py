"""Cuts the fixed train/val/test split of ~/dataset/<name>.parquet with the same protocol
as the ghc2f splits (data/<name>/splits), so other datasets get what all_beauty has:

    1. drops repeated (userId, itemId, timestamp) rows
    2. keeps users with at least MIN_INTER interactions
    3. per user, by timestamp: last interaction -> test, second to last -> val, rest -> train

Timestamp ties (IMDB only has the review date) are broken by the parquet's row order.
Checked against ghc2f: on all_beauty this script rebuilds its splits row for row.

    .venv/bin/python preprocess_data/build_splits.py imdb [out_dir]

Writes train/val/test/trainval.parquet + split_stats.json to out_dir (default
dataset/<name>), where build_fixed_split.py finds it (splits_location.py).
"""
import json
import os
import sys

import pandas as pd

from complement_folds import RAW_DATA_DIR, REPO_ROOT

MIN_INTER = 3


def cut(df):
    df = df.rename(columns={"text": "review"}) if "review" not in df else df
    n_raw = len(df)
    df = df.drop_duplicates(["userId", "itemId", "timestamp"])
    n_dedup = len(df)
    per_user = df.groupby("userId").size()
    df = df[df["userId"].map(per_user) >= MIN_INTER]

    # stable sort: ties keep the parquet's row order
    df = df.sort_values(["userId", "timestamp"], kind="stable").reset_index(drop=True)
    from_end = df.groupby("userId").cumcount(ascending=False)
    splits = {"train": df[from_end >= 2], "val": df[from_end == 1], "test": df[from_end == 0]}
    splits = {k: v.reset_index(drop=True) for k, v in splits.items()}
    splits["trainval"] = df[from_end >= 1].reset_index(drop=True)

    stats = {"raw": {"interactions": n_raw}, "duplicate_user_item_time_rows_dropped": n_raw - n_dedup,
             "users_removed_by_min_inter": int((per_user < MIN_INTER).sum()),
             "interactions_removed_by_min_inter": n_dedup - len(df), "min_inter": MIN_INTER}
    return splits, stats


if __name__ == "__main__":
    name = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO_ROOT, "dataset", name)
    input_file = os.path.join(RAW_DATA_DIR, f"{name}.parquet")
    raw = pd.read_parquet(input_file)

    splits, stats = cut(raw)
    stats["raw"]["users"] = int(raw["userId"].nunique())
    train_items = set(splits["train"]["itemId"])
    stats.update({
        "dataset": name, "input_file": input_file,
        "splits": {k: {"users": int(v["userId"].nunique()), "items": int(v["itemId"].nunique()),
                       "interactions": len(v)} for k, v in splits.items()},
        "frac_val_items_unseen_in_train": float((~splits["val"]["itemId"].isin(train_items)).mean()),
        "frac_test_items_unseen_in_trainval": float((~splits["test"]["itemId"].isin(set(splits["trainval"]["itemId"]))).mean()),
        "frac_users_with_timestamp_tie_at_cut": float(
            (splits["val"]["timestamp"].values == splits["test"]["timestamp"].values).mean()),
    })

    os.makedirs(out_dir, exist_ok=True)
    for k, v in splits.items():
        v.to_parquet(os.path.join(out_dir, f"{k}.parquet"), index=False)
    with open(os.path.join(out_dir, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats["splits"]), f"-> {out_dir}")
