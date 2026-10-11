"""Stage 1 of APH preprocessing: (aspect, sentiment) quadruples of the train reviews of a
fixed split (see preprocess_data/aph/aspect_extraction.py), row-aligned with train.parquet.

Needs spaCy + en_core_web_sm + NLTK WordNet, which live in aspect_cor-rating's venv, not iRev's:

    ~/aspect_cor-rating/.venv/bin/python preprocess_data/extract_aph_aspects.py all_beauty

Only train reviews are read: the hypergraph must not see val/test reviews (the authors'
as_kg.txt was built over the whole dataset). Then build the model files with
preprocess_data/build_aph_fixed_split.py.
"""
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
N_PROCESS = int(os.environ.get("APH_N_PROCESS", 8))


def quadruples_path(dataset_name):
    return os.path.join(REPO_ROOT, "dataset", f"{dataset_name}_data", "fold_0", "train", "aph_quadruples.csv")


if __name__ == "__main__":
    # aqui, não no topo: build_aph_fixed_split.py importa quadruples_path no venv do iRev, sem spaCy
    sys.path.append(SCRIPT_DIR)
    from aph.aspect_extraction import quadruples
    from splits_location import splits_dir as default_splits_dir

    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "all_beauty"
    splits_dir = default_splits_dir(dataset_name)

    texts = pd.read_parquet(os.path.join(splits_dir, "train.parquet"),
                            columns=["review"])["review"].fillna("").astype(str).tolist()
    rows = pd.DataFrame(quadruples(texts, N_PROCESS),
                        columns=["row", "aspect_raw", "aspect", "opinion", "sentiment"])

    out_path = quadruples_path(dataset_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows.to_csv(out_path, index=False)
    print(f"{len(texts)} reviews de treino, {rows['row'].nunique()} com aspecto; "
          f"{len(rows)} quádruplas, {rows['aspect'].nunique()} aspectos; "
          f"polaridade {rows['sentiment'].value_counts().to_dict()} -> {out_path}")
