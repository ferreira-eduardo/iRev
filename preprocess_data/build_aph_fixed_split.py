"""Stage 2 of APH preprocessing: the per-user and per-item hyperedges models/aph.py loads,
built from preprocess_data/extract_aph_aspects.py's train quadruples.

    ~/aspect_cor-rating/.venv/bin/python preprocess_data/extract_aph_aspects.py all_beauty
    .venv/bin/python preprocess_data/build_aph_fixed_split.py all_beauty

Each train review with aspects gives hyperedges (user, item, aspect, sentiment). Written to
fold_0/train:
    aph_user_edges.npy (user_num, E, 3): [aspect, sentiment, item] of each user's hyperedges
    aph_item_edges.npy (item_num, E, 3): [aspect, sentiment, user] of each item's hyperedges
    aph_stats.json: num_aspects, aspect vocabulary
Aspect id 0 is padding (aspects are 1..num_aspects); sentiment ids are 0/1/2 for
negative/neutral/positive. Nodes with more than MAX_EDGES hyperedges keep the most recent.

Ids are remapped with fold_0's user_ids.npy/item_ids.npy and checked against
fold_0/train/Train.npy, so the files cannot disagree with the rest of fold_0.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from build_fixed_split import read_split
from complement_folds import REPO_ROOT, remap_ids
from extract_aph_aspects import quadruples_path
from splits_location import splits_dir as default_splits_dir

MAX_EDGES = int(os.environ.get("APH_MAX_EDGES", 64))
SENTIMENT_ID = {-1: 0, 0: 1, 1: 2}


def edge_table(edges, center_col, other_col, n):
    """(n, E, 3) [aspect, sentiment, other], most recent hyperedges first."""
    table = np.zeros((n, MAX_EDGES, 3), dtype=np.int64)
    edges = edges.sort_values("timestamp", ascending=False, kind="stable")
    edges = edges[edges.groupby(center_col).cumcount() < MAX_EDGES]
    slot = edges.groupby(center_col).cumcount().to_numpy()
    table[edges[center_col].to_numpy(), slot] = edges[["aspect_id", "sentiment_id", other_col]].to_numpy()
    return table


if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "all_beauty"
    splits_dir = default_splits_dir(dataset_name)
    fold_folder = os.path.join(REPO_ROOT, "dataset", f"{dataset_name}_data", "fold_0")
    train_path = os.path.join(fold_folder, "train")

    train = read_split(splits_dir, "train").drop(columns="review_emb")
    quads = pd.read_csv(quadruples_path(dataset_name), keep_default_na=False)
    if len(quads) and quads["row"].max() >= len(train):
        sys.exit("aph_quadruples.csv não bate com train.parquet; rode extract_aph_aspects.py de novo.")

    u_map = {uid: i for i, uid in enumerate(np.load(f"{fold_folder}/user_ids.npy"))}
    i_map = {iid: i for i, iid in enumerate(np.load(f"{fold_folder}/item_ids.npy"))}
    train = remap_ids(train, u_map, i_map)
    if not np.array_equal(np.load(f"{train_path}/Train.npy"), train[["user_id", "item_id"]].values):
        sys.exit(f"split recomputado não bate com {train_path}/Train.npy; rode build_fixed_split.py de novo.")

    vocab = sorted(quads["aspect"].unique())
    aspect_id = {a: i for i, a in enumerate(vocab, 1)}
    edges = quads.join(train[["user_id", "item_id", "timestamp"]].reset_index(drop=True), on="row")
    edges["aspect_id"] = edges["aspect"].map(aspect_id)
    edges["sentiment_id"] = edges["sentiment"].map(SENTIMENT_ID)

    # +2: mesmo tamanho das tabelas por usuário/item do resto da fold (padding)
    user_count, item_count = len(u_map) + 2, len(i_map) + 2
    for side, center, other, n in (("user", "user_id", "item_id", user_count),
                                   ("item", "item_id", "user_id", item_count)):
        table = edge_table(edges, center, other, n)
        np.save(f"{train_path}/aph_{side}_edges.npy", table)
        degree = edges.groupby(center).size()
        print(f"{side}: {degree.size}/{n - 2} com hiperarestas, média {degree.mean():.2f}, "
              f"{int((degree > MAX_EDGES).sum())} truncados em {MAX_EDGES}")

    with open(f"{train_path}/aph_stats.json", "w") as f:
        json.dump({"num_aspects": len(vocab), "max_edges": MAX_EDGES, "aspects": vocab}, f)
    print(f"{len(edges)} hiperarestas, {len(vocab)} aspectos -> {train_path}")
