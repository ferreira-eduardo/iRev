"""
Preprocessing for models/kann.py. We are NOT reproducing the original KANN paper's
datasets/entity-linking pipeline (no entity_index / pretrained TransE .bin files exist
anywhere in this environment -- see add_eduardo/KANN/read_ratings.py, which only loads
those artifacts, never builds them). Instead this script builds a lightweight, dataset-
agnostic substitute end to end:

  1. "Entities" = frequent review tokens (a simple frequency-based vocab per fold,
     computed from TRAIN+VAL only, matching the no-leakage convention already used by
     generate_dataset_letter.py / preprocess_data/complement_folds.py).
  2. "Relations" = {pos, neg} by the same PIVOT convention used for LETTER: two
     entities that co-occur (adjacent, within a review) get a triple whose relation is
     pos/neg depending on that review's rating.
  3. A small from-scratch TransE (margin ranking loss, random negative sampling) is
     trained per fold on those triples -- no new heavy dependency.
  4. models/kann.py's two embedding tables are derived from the trained TransE model:
       kann_entity_ori_emb.npy = the entity embedding table itself
       kann_entity_re_emb.npy  = per-entity mean of concat(head, relation, tail) over
                                 every triple the entity participates in
  5. kann_user_entities.npy / kann_item_entities.npy: padded (dedup, first-seen order)
     sequences of entity ids per user/item, built from TRAIN+VAL reviews.

This is a methodological approximation, not a port of the original paper's KG -- see
the conversation this script was authored in for why (no source KG/entity-linking
artifacts were recoverable).
"""
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# --- CONFIGURATION ---
datasets = ['amazon', 'imdb', 'rotten_tomatoes']
PIVOT = 3
D_MODEL = 50            # must match config.py's DefaultConfig.kann_d_model
ENTITY_SEQ_LEN = 20      # Su / Si: padded entity-sequence length per user/item
MAX_ENTITIES = 3000      # entity vocab cap per fold
MIN_ENTITY_FREQ = 5      # drop tokens rarer than this
MAX_TOKENS_PER_REVIEW = 40  # bound co-occurrence-pair cost for very long reviews

TRANSE_EPOCHS = 5
TRANSE_LR = 1e-2
TRANSE_MARGIN = 1.0
TRANSE_BATCH_SIZE = 1024

# TODO(eduardo): point these at the real source CSV (userId,itemId,rating,review,foldId)
# and matching review-embedding .npy for each dataset (same source as generate_dataset_letter.py).
SOURCE_CSV_DIR = '../../../../dataset'

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "to", "of", "in", "on", "at", "for", "with",
    "as", "by", "it", "its", "not", "no", "so", "too", "very", "can", "will", "just", "than",
    "then", "there", "here", "you", "your", "yours", "i", "me", "my", "we", "our", "he",
    "she", "they", "them", "his", "her", "their", "have", "has", "had", "do", "does", "did",
    "from", "into", "about", "up", "down", "out", "off", "over", "under", "again", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "how", "what", "when", "where", "why", "which", "who", "whom", "would", "could",
    "should", "also", "one", "two", "get", "got", "like",
}
_TOKEN_RE = re.compile(r"[A-Za-z]{3,}")


def tokenize(text):
    if not isinstance(text, str):
        return []
    return [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS]


def build_entity_vocab(token_lists, max_entities, min_freq):
    """token -> entity id, ids start at 1 (0 is padding)."""
    counts = Counter()
    for tokens in token_lists:
        counts.update(set(tokens))  # per-review doc frequency, not raw term frequency

    vocab = {}
    for token, freq in counts.most_common():
        if freq < min_freq:
            break
        if len(vocab) >= max_entities:
            break
        vocab[token] = len(vocab) + 1
    return vocab


def encode_entities(tokens, vocab, max_len=None):
    """Dedup keeping first-seen order, map through vocab, drop OOV tokens."""
    seen = []
    seen_set = set()
    for t in tokens:
        eid = vocab.get(t)
        if eid is None or eid in seen_set:
            continue
        seen_set.add(eid)
        seen.append(eid)
        if max_len is not None and len(seen) >= max_len:
            break
    return seen


def pad_ids(ids, seq_len):
    ids = ids[:seq_len]
    return ids + [0] * (seq_len - len(ids))


def build_transe_triples(rows_entities, rows_relation):
    """Adjacent-pair (window=2) co-occurrence triples within each row's entity list."""
    triples = []
    for entities, rel in zip(rows_entities, rows_relation):
        entities = entities[:MAX_TOKENS_PER_REVIEW]
        for a, b in zip(entities, entities[1:]):
            if a == b:
                continue
            triples.append((a, rel, b))
            triples.append((b, rel, a))
    return triples


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


def train_transe(num_entities, num_relations, triples, d_model, device):
    model = _TransE(num_entities, num_relations, d_model).to(device)
    if not triples:
        print("  [TransE] no triples for this fold, keeping random init.")
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=TRANSE_LR)
    triples_t = torch.as_tensor(triples, dtype=torch.long, device=device)  # (N, 3)

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

    return model


def build_re_emb(model, triples, num_entities, d_model, device):
    """Per-entity mean of concat(head, relation, tail) over every triple it appears in."""
    re_emb = np.zeros((num_entities + 1, 3 * d_model), dtype=np.float32)
    if not triples:
        return re_emb

    entity_w = model.entity_emb.weight.detach().cpu().numpy()
    relation_w = model.relation_emb.weight.detach().cpu().numpy()

    sums = np.zeros((num_entities + 1, 3 * d_model), dtype=np.float64)
    counts = np.zeros(num_entities + 1, dtype=np.int64)

    for h, r, t in triples:
        concat = np.concatenate([entity_w[h], relation_w[r], entity_w[t]])
        for entity_id in (h, t):
            sums[entity_id] += concat
            counts[entity_id] += 1

    has_triples = counts > 0
    re_emb[has_triples] = (sums[has_triples] / counts[has_triples, None]).astype(np.float32)
    return re_emb


def build_entity_sequences(context_df, id_col, token_col, vocab, num_total, seq_len):
    matrix = np.zeros((num_total, seq_len), dtype=np.int64)
    grouped = context_df.groupby(id_col)[token_col].apply(lambda lists: [tok for l in lists for tok in l])
    for idx, tokens in grouped.items():
        matrix[idx] = pad_ids(encode_entities(tokens, vocab, max_len=seq_len), seq_len)
    return matrix


for DATASET_NAME in datasets:
    print(f"\n{'=' * 40}\nDATASET: {DATASET_NAME}\n{'=' * 40}")
    BASE_OUTPUT_DIR = f'./dataset/{DATASET_NAME}'
    df = pd.read_csv(f'{SOURCE_CSV_DIR}/{DATASET_NAME}.csv')

    user_map = {u: i for i, u in enumerate(df['userId'].unique())}
    item_map = {i: x for x, i in enumerate(df['itemId'].unique())}
    df['u_idx'] = df['userId'].map(user_map).astype(int)
    df['i_idx'] = df['itemId'].map(item_map).astype(int)

    num_users = len(user_map)
    num_items = len(item_map)
    print(f"Users: {num_users}, Items: {num_items}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    folds = sorted(df['foldId'].unique())
    for k in folds:
        print(f"--- Processing FOLD {k} ---")

        FOLD_DIR = os.path.join(BASE_OUTPUT_DIR, f'fold_{k}')
        TRAIN_DIR = os.path.join(FOLD_DIR, 'train')
        os.makedirs(TRAIN_DIR, exist_ok=True)

        test_df = df[df['foldId'] == k].copy()
        train_full_df = df[df['foldId'] != k].copy()
        val_df = train_full_df.sample(frac=0.125, random_state=42)
        train_df = train_full_df.drop(val_df.index)

        # Train+val only: same no-leakage convention as generate_dataset_letter.py.
        context_df = pd.concat([train_df, val_df])
        context_df['tokens'] = context_df['review'].apply(tokenize)
        # relation ids must be dense 0/1 indices for nn.Embedding(num_relations, ...):
        # 0 = pos (rating > PIVOT), 1 = neg (rating <= PIVOT)
        context_df['relation'] = np.where(context_df['rating'] > PIVOT, 0, 1)

        vocab = build_entity_vocab(context_df['tokens'], MAX_ENTITIES, MIN_ENTITY_FREQ)
        num_entities = len(vocab)
        print(f"  entities: {num_entities}")

        context_df['entity_ids'] = context_df['tokens'].apply(
            lambda toks: encode_entities(toks, vocab, max_len=MAX_TOKENS_PER_REVIEW))

        triples = build_transe_triples(context_df['entity_ids'].tolist(), context_df['relation'].tolist())
        print(f"  triples: {len(triples)}")

        transe = train_transe(num_entities, num_relations=2, triples=triples, d_model=D_MODEL, device=device)

        ori_emb = transe.entity_emb.weight.detach().cpu().numpy().astype(np.float32)
        re_emb = build_re_emb(transe, triples, num_entities, D_MODEL, device)

        np.save(os.path.join(TRAIN_DIR, 'kann_entity_ori_emb.npy'), ori_emb)
        np.save(os.path.join(TRAIN_DIR, 'kann_entity_re_emb.npy'), re_emb)

        user_entities = build_entity_sequences(context_df, 'u_idx', 'entity_ids', vocab, num_users, ENTITY_SEQ_LEN)
        item_entities = build_entity_sequences(context_df, 'i_idx', 'entity_ids', vocab, num_items, ENTITY_SEQ_LEN)

        np.save(os.path.join(TRAIN_DIR, 'kann_user_entities.npy'), user_entities)
        np.save(os.path.join(TRAIN_DIR, 'kann_item_entities.npy'), item_entities)

        # Merge into config_stats.json rather than overwrite (generate_dataset_letter.py
        # may have already written user_num/item_num for this same fold).
        stats_path = os.path.join(FOLD_DIR, 'config_stats.json')
        stats = {}
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
        stats['user_num'] = num_users
        stats['item_num'] = num_items
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)

    print(f"Generation of KANN artifacts for {DATASET_NAME} completed successfully!")

print("\nAll datasets done.")
