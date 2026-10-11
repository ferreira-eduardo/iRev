"""Encodes RLMRec's LLM profiles (generate_rlmrec_profiles.py) into the semantic
embeddings models/rlmrec.py aligns with: rlmrec_{user,item}_prf_emb.npy, row-aligned
with fold_0's user/item ids.

    ~/PycharmProjects/ghc2f/.venv/bin/python preprocess_data/build_rlmrec_fixed_split.py all_beauty

RLMRec encodes the "summarization" field with text-embedding-ada-002; here it is
intfloat/e5-base-v2 (local, already used for LETTER), L2-normalized like ada's vectors.
Rows without a profile (padding ids, or an unparseable "None") stay zero.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_NAME = "intfloat/e5-base-v2"

if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "all_beauty"
    fold_folder = os.path.join(REPO_ROOT, "dataset", f"{dataset_name}_data", "fold_0")
    train_path = os.path.join(fold_folder, "train")
    model = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")

    for side in ("user", "item"):
        # ids crus da split (userId/itemId) -> posição no fold, como remap_ids; +2 linhas de padding
        ids = np.load(f"{fold_folder}/{side}_ids.npy")
        pos = {raw: i for i, raw in enumerate(ids)}
        files = sorted(glob.glob(f"{train_path}/rlmrec_profiles/{side}_*.parquet"))
        if not files:
            sys.exit(f"sem perfis em {train_path}/rlmrec_profiles; rode generate_rlmrec_profiles.py antes.")
        prf = pd.concat([pd.read_parquet(f) for f in files])
        prf = prf[prf["id"].isin(pos) & (prf["summarization"].str.strip().str.lower() != "none")]
        missing = len(ids) - len(prf)

        # e5: prefixo "query: " para similaridade simétrica (model card)
        emb = model.encode(["query: " + s for s in prf["summarization"]], batch_size=128,
                           normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        table = np.zeros((len(ids) + 2, emb.shape[1]), dtype=np.float32)
        table[[pos[i] for i in prf["id"]]] = emb
        np.save(f"{train_path}/rlmrec_{side}_prf_emb.npy", table)
        print(f"{side}: {len(prf)} perfis, {missing} sem perfil -> {train_path}/rlmrec_{side}_prf_emb.npy")
