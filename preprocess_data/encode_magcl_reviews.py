"""Encodes the TRAIN reviews of a fixed split with Sentence-BERT all-MiniLM-L6-v2, the
encoder MAGCL's data_preprocessing.py uses. One .npy row-aligned with train.parquet.

Only train: MAGCL's user-user / item-item review graphs must not see val/test text.
Needs `sentence-transformers`, which lives in ghc2f's venv, not iRev's:

    ~/PycharmProjects/ghc2f/.venv/bin/python preprocess_data/encode_magcl_reviews.py all_beauty

Then build the graphs with preprocess_data/build_magcl_fixed_split.py.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from splits_location import splits_dir as default_splits_dir

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(REPO_ROOT, "embeddings_reviews")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "all_beauty"
    splits_dir = default_splits_dir(dataset_name)
    os.makedirs(OUT_DIR, exist_ok=True)

    texts = pd.read_parquet(os.path.join(splits_dir, "train.parquet"),
                            columns=["review"])["review"].fillna("").astype(str).tolist()
    model = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
    # unnormalized, as in MAGCL (generate_vectors sums the raw S-BERT vectors)
    emb = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    emb[[not t.strip() for t in texts]] = 0.0  # reviews vazias ficam com vetor zero

    out_path = os.path.join(OUT_DIR, f"{dataset_name}_train_minilm.npy")
    np.save(out_path, emb.astype(np.float32))
    print(f"train: {emb.shape} -> {out_path}")
