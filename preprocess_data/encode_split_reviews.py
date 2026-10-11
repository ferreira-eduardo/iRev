"""Encodes the review text of a fixed train/val split with ghc2f's TextEncoder
(intfloat/e5-base-v2, 768-d), so LETTER gets the same review representation as
the ghc2f model. One .npy per split, row-aligned with <split>.parquet.

Needs `transformers`, which lives in ghc2f's venv, not iRev's:

    ~/PycharmProjects/ghc2f/.venv/bin/python preprocess_data/encode_split_reviews.py all_beauty

Then build the LETTER files with preprocess_data/build_letter_fixed_split.py.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
GHC2F_ROOT = os.environ.get("GHC2F_ROOT", "/home/eduardo/PycharmProjects/ghc2f")
OUT_DIR = os.path.join(REPO_ROOT, "embeddings_reviews")

sys.path.append(GHC2F_ROOT)
from model_new_review_aware.text_encoder import TextEncoder  # noqa: E402
from splits_location import splits_dir as default_splits_dir  # noqa: E402

if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "all_beauty"
    splits_dir = default_splits_dir(dataset_name)
    os.makedirs(OUT_DIR, exist_ok=True)

    encoder = TextEncoder().to("cuda" if torch.cuda.is_available() else "cpu")
    # LETTER only pools train+val reviews (save_letter_files never reads test)
    for split in ("train", "val"):
        texts = pd.read_parquet(os.path.join(splits_dir, f"{split}.parquet"),
                                columns=["review"])["review"].fillna("").astype(str)
        non_empty = [t for t in texts if t.strip()]
        encoder.warm_up(non_empty)
        # reviews vazias ficam com vetor zero
        emb = np.stack([encoder.cache[t].float().numpy() if t.strip()
                        else np.zeros(encoder.dim, dtype=np.float32) for t in texts])
        out_path = os.path.join(OUT_DIR, f"{dataset_name}_{split}.npy")
        np.save(out_path, emb.astype(np.float32))
        print(f"{split}: {emb.shape} ({len(texts) - len(non_empty)} vazias) -> {out_path}")
    print(f"truncadas em {encoder.max_length} tokens: {encoder.n_truncated}")
