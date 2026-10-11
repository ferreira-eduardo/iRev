"""
Regenerates the two sets of per-fold artifacts models/letter.py and models/kann.py
need that are missing for dataset/amazon_data/fold_0/train (see logs/grid_search/
Amazon_data_LETTER.log and Amazon_data_KANN.log): Rating_Matrix.npy and the
kann_entity_*_emb.npy / kann_*_entities.npy quartet.

The scripts that were meant to produce these (generate_dataset_letter.py,
generate_dataset_kann.py in this directory) read from a raw amazon.csv +
embeddings_reviews/amazon.npy that don't exist anywhere reachable for this
dataset's actual 591,950-row source (the only amazon.npy found on this machine,
in a sibling project, has 59,701 rows and isn't alignable). preprocess_data/
complement_folds.py -- the canonical pipeline -- explicitly skips both artifacts
for Amazon because its dense-matrix guard (500M cells) is well under Amazon's
actual user_num*item_num (~1.5B cells); see the WARNING branch in that file.

This script sources everything from what's already on disk for fold_0/train
(Train.npy/Train_Score.npy, userReview2Index.npy/itemReview2Index.npy,
config_stats.json) instead -- no raw text or external embeddings needed:

  Rating_Matrix.npy: dense (user_num, item_num) matrix filled directly from
    Train.npy's (user_id, item_id) pairs and Train_Score.npy's per-interaction
    score. Train_Score.npy is already z-scored (see adapt_storage_loocv.py),
    which is fine here -- models/letter.py only uses this matrix to build a
    cosine-similarity KNN graph, it doesn't require a literal 1-5 scale.

  KANN entity files: same "lightweight, dataset-agnostic substitute" approach
    as generate_dataset_kann.py (frequent tokens = entities, adjacency within
    a review = a triple, a small from-scratch TransE), but with entities drawn
    directly from the already-tokenized, already-leakage-free
    userReview2Index.npy/itemReview2Index.npy (vocab-index ids) instead of
    re-tokenizing raw review text. Unlike that script, only a single
    synthetic relation is used (no rating-based pos/neg split): Train.npy
    doesn't preserve which specific review each interaction's tokens belong
    to inside the per-user/item-aggregated userReview2Index.npy, so a
    per-triple pos/neg label can't be reconstructed reliably from what's on
    disk. models/kann.py never reads the relation count back (the relation
    table is only used internally here, to build kann_entity_re_emb.npy's
    fixed-width concat), so this doesn't change its output shape or how
    kann.py loads it.

    D_MODEL is pinned to 50 to match config.py's DefaultConfig.kann_d_model
    default and grid_search.py's PARAM_GRIDS["KANN"]["kann_d_model"] (which is
    itself pinned to [50] for exactly this reason -- see grid_search.py's
    kann_d_model note).

Only fold_0 (matches config.py's DefaultConfig.folds_train = [0], what
grid_search.py trains on by default). Rating_Matrix.npy alone can be several
GB for a large/dense dataset; do not casually loop this over all 5 folds
without checking disk headroom first.

Reused across datasets (originally written for Amazon_data, later also run
for TripAdvisor_data -- see logs/grid_search/TripAdvisor_data_KANN.log): pass
--dataset_dir=<name>_data. Each artifact is only (re)built if missing, so
running this against a dataset that already has a real Rating_Matrix.npy
(e.g. TripAdvisor_data, built by preprocess_data/complement_folds.py from
actual review data) never clobbers it with this script's cruder proxy.

Usage:
    python add_eduardo/regen_amazon_fold0_letter_kann.py --dataset_dir=tripadvisor_data
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

D_MODEL = 50
ENTITY_SEQ_LEN = 20
MAX_ENTITIES = 3000
MIN_ENTITY_FREQ = 5
MAX_TOKENS_PER_REVIEW = 40

TRANSE_EPOCHS = 5
TRANSE_LR = 1e-2
TRANSE_MARGIN = 1.0
TRANSE_BATCH_SIZE = 1024


def load_stats():
    with open(f"{FOLD_DIR}/config_stats.json") as f:
        return json.load(f)


def build_rating_matrix(stats, force=False):
    if os.path.exists(f"{TRAIN_DIR}/Rating_Matrix.npy") and not force:
        print("Rating_Matrix.npy already exists, skipping.")
        return
    user_num, item_num = stats["user_num"], stats["item_num"]
    train = np.load(f"{TRAIN_DIR}/Train.npy")
    scores = np.load(f"{TRAIN_DIR}/Train_Score.npy")

    matrix = np.zeros((user_num, item_num), dtype=np.float32)
    matrix[train[:, 0], train[:, 1]] = scores.astype(np.float32)
    np.save(f"{TRAIN_DIR}/Rating_Matrix.npy", matrix)
    print(f"Rating_Matrix.npy: shape={matrix.shape} "
          f"size={matrix.nbytes / 1e9:.2f}GB nonzero={np.count_nonzero(matrix)}")


class _TransE(nn.Module):
    def __init__(self, num_entities, num_relations, d_model):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities + 1, d_model, padding_idx=0)
        self.relation_emb = nn.Embedding(num_relations, d_model)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        with torch.no_grad():
            self.entity_emb.weight[0].zero_()

    def score(self, h, r, t):
        return torch.norm(self.entity_emb(h) + self.relation_emb(r) - self.entity_emb(t), p=2, dim=-1)


def build_kann_artifacts(stats, force=False):
    if os.path.exists(f"{TRAIN_DIR}/kann_entity_re_emb.npy") and not force:
        print("kann_entity_re_emb.npy already exists, skipping KANN artifacts.")
        return
    user_num, item_num = stats["user_num"], stats["item_num"]
    user_reviews = np.load(f"{TRAIN_DIR}/userReview2Index.npy")
    item_reviews = np.load(f"{TRAIN_DIR}/itemReview2Index.npy")

    def review_rows(arr):
        flat = arr.reshape(-1, arr.shape[-1])
        for row in flat:
            if row.any():
                yield row[:MAX_TOKENS_PER_REVIEW]

    all_rows = list(review_rows(user_reviews)) + list(review_rows(item_reviews))

    counts = Counter()
    for row in all_rows:
        counts.update(set(int(t) for t in row if t != 0))

    vocab = {}
    for token, freq in counts.most_common():
        if freq < MIN_ENTITY_FREQ:
            break
        if len(vocab) >= MAX_ENTITIES:
            break
        vocab[token] = len(vocab) + 1
    num_entities = len(vocab)
    print(f"entities: {num_entities}")

    def encode(tokens):
        seen, seen_set = [], set()
        for t in tokens:
            eid = vocab.get(int(t))
            if eid is None or eid in seen_set:
                continue
            seen_set.add(eid)
            seen.append(eid)
        return seen

    triples = []
    for row in all_rows:
        ents = encode(row)
        for a, b in zip(ents, ents[1:]):
            if a == b:
                continue
            triples.append((a, 0, b))
            triples.append((b, 0, a))
    print(f"triples: {len(triples)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _TransE(num_entities, num_relations=1, d_model=D_MODEL).to(device)

    if triples:
        optimizer = torch.optim.Adam(model.parameters(), lr=TRANSE_LR)
        triples_t = torch.as_tensor(triples, dtype=torch.long, device=device)
        for epoch in range(TRANSE_EPOCHS):
            perm = triples_t[torch.randperm(triples_t.size(0), device=device)]
            total_loss = 0.0
            for start in range(0, perm.size(0), TRANSE_BATCH_SIZE):
                batch = perm[start:start + TRANSE_BATCH_SIZE]
                h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
                neg_t = torch.randint(1, num_entities + 1, t.shape, device=device)
                pos_dist = model.score(h, r, t)
                neg_dist = model.score(h, r, neg_t)
                loss = torch.clamp(TRANSE_MARGIN + pos_dist - neg_dist, min=0).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.size(0)
            print(f"  [TransE] epoch {epoch + 1}/{TRANSE_EPOCHS} loss={total_loss / perm.size(0):.4f}")
    else:
        print("  [TransE] no triples, keeping random init.")

    ori_emb = model.entity_emb.weight.detach().cpu().numpy().astype(np.float32)

    re_emb = np.zeros((num_entities + 1, 3 * D_MODEL), dtype=np.float32)
    if triples:
        entity_w = ori_emb
        relation_w = model.relation_emb.weight.detach().cpu().numpy()
        sums = np.zeros((num_entities + 1, 3 * D_MODEL), dtype=np.float64)
        cnts = np.zeros(num_entities + 1, dtype=np.int64)
        for h, r, t in triples:
            concat = np.concatenate([entity_w[h], relation_w[r], entity_w[t]])
            for eid in (h, t):
                sums[eid] += concat
                cnts[eid] += 1
        has_triples = cnts > 0
        re_emb[has_triples] = (sums[has_triples] / cnts[has_triples, None]).astype(np.float32)

    # models/kann.py's _key_padding_mask treats entity id 0 as padding and feeds
    # it straight to nn.MultiheadAttention; a user/item whose review tokens never
    # cleared MIN_ENTITY_FREQ/MAX_ENTITIES ends up with an all-zero (fully masked)
    # sequence, and softmax over an all-masked row is NaN -- silently poisons the
    # loss (observed: Val MSE: nan). Reserve one extra, always-zero-embedding
    # entity id as a neutral fallback so no sequence is ever fully padding.
    unk_id = num_entities + 1
    ori_emb = np.concatenate([ori_emb, np.zeros((1, D_MODEL), dtype=np.float32)], axis=0)
    re_emb = np.concatenate([re_emb, np.zeros((1, 3 * D_MODEL), dtype=np.float32)], axis=0)

    np.save(f"{TRAIN_DIR}/kann_entity_ori_emb.npy", ori_emb)
    np.save(f"{TRAIN_DIR}/kann_entity_re_emb.npy", re_emb)

    def build_sequences(arr, num_total):
        matrix = np.zeros((num_total, ENTITY_SEQ_LEN), dtype=np.int64)
        n = min(arr.shape[0], num_total)
        flat = arr[:n].reshape(n, -1)
        for i in range(n):
            ents = encode(flat[i])[:ENTITY_SEQ_LEN] or [unk_id]
            matrix[i, :len(ents)] = ents
        # rows past arr.shape[0] (padding ids beyond what userReview2Index/
        # itemReview2Index cover) are still all-zero here -- same fallback.
        empty_rows = matrix.sum(axis=1) == 0
        matrix[empty_rows, 0] = unk_id
        return matrix

    kann_user_entities = build_sequences(user_reviews, user_num)
    kann_item_entities = build_sequences(item_reviews, item_num)
    np.save(f"{TRAIN_DIR}/kann_user_entities.npy", kann_user_entities)
    np.save(f"{TRAIN_DIR}/kann_item_entities.npy", kann_item_entities)
    print(f"kann_user_entities.npy: {kann_user_entities.shape}, "
          f"kann_item_entities.npy: {kann_item_entities.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="amazon_data",
                         help="e.g. amazon_data, tripadvisor_data (config.py's setting_path)")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                         help="rebuild even if the target artifact already exists")
    args = parser.parse_args()

    FOLD_DIR = f"dataset/{args.dataset_dir}/fold_{args.fold}"
    TRAIN_DIR = f"{FOLD_DIR}/train"

    stats = load_stats()
    build_rating_matrix(stats, force=args.force)
    build_kann_artifacts(stats, force=args.force)
    print("Done.")
