import math
import random
from typing import List, Set
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

def get_ground_truth(df, user_id, positive_threshold):
    # Retorna conjunto de itens que o usuário avaliou positivamente
    subset = df[(df['userId'] == user_id) & (df['rating'] >= positive_threshold)]
    return set(subset['itemId'].tolist())


def get_user_history(df, user_id):
    # Retorna todos os itens que o usuário interagiu (para serendipity/filtragem)
    return set(df[df['userId'] == user_id]['itemId'].tolist())

def create_eval_pool(
        ground_truth: list[int],
        all_items_ids: Set[int],
        user_interacted_items: Set[int],
        num_negatives: int = 100) -> torch.Tensor:
    """
    pool of items merging items from ground truth with aleatory items.
    """
    # Items not interact from user
    negative_items_pool = all_items_ids - user_interacted_items

    # Amostra aleatoriamente 'num_negativos' itens do pool negativo
    num_to_sample = min(len(negative_items_pool), num_negatives)
    negative_samples = np.random.choice(list(negative_items_pool), num_to_sample, replace=False)

    # Junta os itens positivos com os negativos amostrados
    final_pool = list(ground_truth) + list(negative_samples)
    np.random.shuffle(final_pool)
    return torch.tensor(final_pool, dtype=torch.long)


def item_popularity(train_df: pd.DataFrame) -> dict[int, float]:
    """
    Calculates the popularity of each item based on interaction counts.
    """
    # Count how many interactions each item has
    interaction_counts = train_df['itemId'].value_counts()
    total_interactions = len(train_df)

    # Calculate popularity as a fraction of total interactions
    item_popularity = interaction_counts / total_interactions

    return item_popularity.to_dict()


def calculate_ndcg_at_k(recommended_list, ground_truth, k):
    """
    nDCG@k com relevância binária:
    - recommended_list: lista ordenada de item_ids recomendados
    - ground_truth: conjunto ou lista de itens relevantes
    - k: cutoff
    """
    if not ground_truth:
        return 0.0

    # Garante que ground_truth é um set (membership O(1))
    if not isinstance(ground_truth, set):
        ground_truth = set(ground_truth)

    top_k = recommended_list[:k]
    if len(top_k) == 0:
        return 0.0

    # Ganhos binários: 1 se o item está no ground_truth, 0 caso contrário
    gains = np.fromiter(
        (1.0 if item in ground_truth else 0.0 for item in top_k),
        dtype=np.float32,
        count=len(top_k)
    )

    # Descontos: 1/log2(pos+1), com posição começando em 1
    # i = 0 → posição 1 → log2(2), etc.
    discounts = 1.0 / np.log2(np.arange(2, len(top_k) + 2))

    dcg = float((gains * discounts).sum())

    # IDCG: melhor caso possível → todos os relevantes no topo
    ideal_len = min(k, len(ground_truth))
    if ideal_len == 0:
        return 0.0

    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_len + 2))
    idcg = float(ideal_discounts.sum())

    return dcg / idcg if idcg > 0.0 else 0.0


def calculate_recall_at_k(recommended_list, ground_truth, k):
    if len(ground_truth) == 0: return 0.0
    hits = len(set(recommended_list[:k]).intersection(ground_truth))
    return hits / len(ground_truth)


def calculate_mrr_at_k(recommended_list, ground_truth, k):
    for i, item_id in enumerate(recommended_list[:k]):
        if item_id in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def sample_negatives(pos_set, all_items, num_neg):
    negatives = []
    while len(negatives) < num_neg:
        neg_id = random.sample(all_items, 1)[0]
        if neg_id not in pos_set:
            negatives.append(neg_id)
    return negatives

def calculate_map_at_k(recommended_list, ground_truth, k):
    if not ground_truth:
        return 0.0

    if not isinstance(ground_truth, set):
        ground_truth = set(ground_truth)

    hits = 0
    sum_precisions = 0.0

    for i, item_id in enumerate(recommended_list[:k], start=1):
        if item_id in ground_truth:
            hits += 1
            precision_at_i = hits / i
            sum_precisions += precision_at_i

    return sum_precisions / len(ground_truth) if len(ground_truth) > 0 else 0.0

def calculate_serendipity(model, ground_truth, user_history, recommended_list, k):
    """
    Serendipity@k:
    - Consider only recommended items that are also in ground_truth.
    - For each such item, compute its average cosine similarity to the user's history items.
    - Serendipity is 1 - avg_similarity (more dissimilar = more serendipitous).
    - Finally, average this "surprise" over all relevant recommended items.
    """
    # nothing to compute
    if not user_history or not ground_truth:
        return 0.0

    device = model.iid_embedding.weight.device
    item_embs = model.iid_embedding.weight  # [num_items, emb_dim]

    # top-k recommended items
    rec_k = recommended_list[:k]

    # ensure ground_truth is a set for O(1) membership
    if not isinstance(ground_truth, set):
        ground_truth = set(ground_truth)

    # we only consider recommended items that are also relevant
    relevant_recs = [i for i in rec_k if i in ground_truth]
    if len(relevant_recs) == 0:
        return 0.0

    # embeddings for user's history items
    hist_ids = torch.tensor(list(user_history), dtype=torch.long, device=device)
    if hist_ids.numel() == 0:
        return 0.0  # defensive, though we already checked user_history

    hist_embs = item_embs[hist_ids]  # [H, D]

    # embeddings for relevant recommended items
    rec_ids = torch.tensor(relevant_recs, dtype=torch.long, device=device)
    rec_embs = item_embs[rec_ids]    # [R, D]

    # normalize for cosine similarity
    hist_embs = F.normalize(hist_embs, p=2, dim=1)  # [H, D]
    rec_embs  = F.normalize(rec_embs,  p=2, dim=1)  # [R, D]

    # pairwise cosine similarities: [H, R]
    # (each column j: similarity of rec_j with all history items)
    sim_mat = hist_embs @ rec_embs.t()  # [H, R]

    # average similarity per recommended item (over the history)
    mean_sim_per_rec = sim_mat.mean(dim=0)  # [R]

    # "surprise" = 1 - similarity
    surprise_per_rec = 1.0 - mean_sim_per_rec  # [R]

    # final serendipity: média dos itens recomendados relevantes
    serendipity = surprise_per_rec.mean().item()
    return float(serendipity)



def calculate_diversity(model, recommended_list, k):
    device = model.iid_embedding.weight.device
    item_embs = model.iid_embedding.weight

    rec_k = recommended_list[:k]
    if len(rec_k) < 2:
        return 0.0

    ids_tensor = torch.tensor(rec_k, dtype=torch.long, device=device)
    emb_recs = item_embs[ids_tensor]
    emb_recs = F.normalize(emb_recs, p=2, dim=1)

    sim_mat = emb_recs @ emb_recs.t()           # (K, K)
    K = sim_mat.size(0)

    # soma somente acima da diagonal (pares i<j)
    upper = torch.triu(sim_mat, diagonal=1)
    num_pairs = K * (K - 1) / 2
    avg_sim = upper.sum() / num_pairs

    diversity = 1.0 - avg_sim.item()
    return diversity
