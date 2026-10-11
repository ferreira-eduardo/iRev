"""Builds MAGCL's review-similarity graphs (magcl_{user,item}_graph.npz) for a
fixed-split dataset built by build_fixed_split.py.

    ~/PycharmProjects/ghc2f/.venv/bin/python preprocess_data/encode_magcl_reviews.py all_beauty
    .venv/bin/python preprocess_data/build_magcl_fixed_split.py all_beauty

As MAGCL's data_preprocessing.py: a user's (item's) document is the SUM of the
all-MiniLM-L6-v2 vectors of its reviews, and two users (items) are linked when the
cosine of their documents is > 0.83. Differences, both needed at Amazon scale:
    - train reviews only (the original pools every review, val/test included);
    - cosines in GPU/CPU blocks into a sparse matrix, instead of a Python double
      loop over a dense n x n matrix (Musical Instruments: 82k x 82k users).

Ids are remapped with fold_0's saved user_ids.npy/item_ids.npy and checked
against fold_0/train/Train.npy, as build_letter_fixed_split.py does.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

from build_fixed_split import read_split
from complement_folds import REPO_ROOT, remap_ids
from splits_location import splits_dir as default_splits_dir

EMB_DIR = os.path.join(REPO_ROOT, "embeddings_reviews")
SIM_THRESHOLD = 0.83  # MAGCL's Matrix_sim
BLOCK = 4096


def similarity_graph(doc, threshold=SIM_THRESHOLD):
    """Binary symmetric graph of cos(doc_a, doc_b) > threshold, a != b; zero rows stay isolated."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.nn.functional.normalize(torch.from_numpy(doc).to(device), dim=1)
    rows, cols = [], []
    for a in range(0, len(x), BLOCK):
        r, c = torch.nonzero(x[a:a + BLOCK] @ x.T > threshold, as_tuple=True)
        r += a
        keep = r != c
        rows.append(r[keep].cpu().numpy())
        cols.append(c[keep].cpu().numpy())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(doc), len(doc)))


if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "all_beauty"
    splits_dir = default_splits_dir(dataset_name)
    fold_folder = os.path.join(REPO_ROOT, "dataset", f"{dataset_name}_data", "fold_0")
    train_path = os.path.join(fold_folder, "train")

    train = read_split(splits_dir, "train")
    emb = np.load(os.path.join(EMB_DIR, f"{dataset_name}_train_minilm.npy"))
    if len(emb) != len(train):
        sys.exit(f"train: {len(emb)} embeddings para {len(train)} linhas; rode encode_magcl_reviews.py de novo.")

    u_map = {uid: i for i, uid in enumerate(np.load(f"{fold_folder}/user_ids.npy"))}
    i_map = {iid: i for i, iid in enumerate(np.load(f"{fold_folder}/item_ids.npy"))}
    train = remap_ids(train, u_map, i_map)
    if not np.array_equal(np.load(f"{train_path}/Train.npy"), train[["user_id", "item_id"]].values):
        sys.exit(f"split recomputado não bate com {train_path}/Train.npy; rode build_fixed_split.py de novo.")

    # +2: same padding rows as the rest of fold_0 (config_stats user_num/item_num)
    for side, col, n in (("user", "user_id", len(u_map) + 2), ("item", "item_id", len(i_map) + 2)):
        doc = np.zeros((n, emb.shape[1]), dtype=np.float32)
        np.add.at(doc, train[col].to_numpy(), emb)
        graph = similarity_graph(doc)
        sp.save_npz(f"{train_path}/magcl_{side}_graph.npz", graph)
        print(f"{side}: {n} nós, {graph.nnz} arestas ({graph.nnz / n:.1f} por nó) "
              f"-> {train_path}/magcl_{side}_graph.npz")
