"""Where a dataset's fixed train/val/test splits live: $IREV_SPLITS_DIR when set, else
dataset/<name>/splits (all_beauty) or dataset/<name> (imdb, musical_instruments)."""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def splits_dir(name):
    if os.environ.get("IREV_SPLITS_DIR"):
        return os.environ["IREV_SPLITS_DIR"]
    base = os.path.join(REPO_ROOT, "dataset", name)
    nested = os.path.join(base, "splits")
    return nested if os.path.isfile(os.path.join(nested, "train.parquet")) else base
