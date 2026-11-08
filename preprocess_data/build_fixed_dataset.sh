#!/usr/bin/env bash
# Builds dataset/<name>_data/fold_0 from ~/dataset/<name>.parquet with every model's
# inputs (the all_beauty pipeline, for any dataset):
#
#     bash preprocess_data/build_fixed_dataset.sh imdb
#
# Splits: dataset/<name>/splits or dataset/<name> (preprocess_data/splits_location.py);
# cut here with build_splits.py when neither exists. Run from the repo root.
set -euo pipefail

NAME=$1
IREV_PY=.venv/bin/python
GHC2F_PY=${GHC2F_PY:-$HOME/PycharmProjects/ghc2f/.venv/bin/python}     # transformers (LETTER)
SPACY_PY=${SPACY_PY:-$HOME/aspect_cor-rating/.venv/bin/python}         # spaCy + WordNet (APH)

if [ -z "${IREV_SPLITS_DIR:-}" ] && [ ! -f "dataset/$NAME/splits/train.parquet" ] && [ ! -f "dataset/$NAME/train.parquet" ]; then
    echo "== splits -> dataset/$NAME"
    (cd preprocess_data && ../$IREV_PY build_splits.py "$NAME" "$PWD/../dataset/$NAME")
fi
echo "== splits from $($IREV_PY -c "import sys; sys.path.insert(0, 'preprocess_data'); from splits_location import splits_dir; print(splits_dir('$NAME'))")"

echo "== iRev files (ids, ratings, reviews, w2v)"
(cd preprocess_data && ../$IREV_PY build_fixed_split.py "$NAME")

echo "== LETTER: review embeddings"
$GHC2F_PY preprocess_data/encode_split_reviews.py "$NAME"
(cd preprocess_data && ../$IREV_PY build_letter_fixed_split.py "$NAME")

echo "== KANN"
$IREV_PY add_eduardo/regen_amazon_fold0_letter_kann.py --dataset_dir="${NAME}_data" --force

echo "== APH"
$SPACY_PY preprocess_data/extract_aph_aspects.py "$NAME"
(cd preprocess_data && ../$IREV_PY build_aph_fixed_split.py "$NAME")

echo "== MAGCL: review-similarity graphs"
$GHC2F_PY preprocess_data/encode_magcl_reviews.py "$NAME"
(cd preprocess_data && ../$IREV_PY build_magcl_fixed_split.py "$NAME")
