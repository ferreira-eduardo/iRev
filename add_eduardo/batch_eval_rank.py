import numpy as np
import torch
from tqdm import tqdm

from main import sample_negatives, unpack_input


def batched_ranking_eval(
    model, opt, user_pos, all_item_ids, num_neg, K, max_valid_item,
    chunk_pairs=50000
):
    model.eval()

    users_meta = []   # [(start, end, candidates, pos_items), ...]
    pair_users = []
    pair_items = []

    # 1) Build (user,item) pairs for ALL users
    for user, pos_items in tqdm(user_pos.items(), total=len(user_pos), desc="Build pairs"):
        pos_items = [i for i in pos_items if 0 <= i < max_valid_item]
        if not pos_items:
            continue

        pos_set = set(pos_items)
        neg_items = sample_negatives(user, pos_set, all_item_ids, num_neg=num_neg)

        candidates = list(pos_set) + list(neg_items)
        candidates = [i for i in candidates if 0 <= i < max_valid_item]
        if not candidates:
            continue

        start = len(pair_users)
        pair_users.extend([user] * len(candidates))
        pair_items.extend(candidates)
        end = len(pair_users)

        users_meta.append((start, end, candidates, list(pos_set)))

    if not users_meta:
        return None

    # 2) Run inference in big chunks
    all_scores = np.empty(len(pair_users), dtype=np.float32)

    device = next(model.parameters()).device
    with torch.inference_mode():  # slightly better than no_grad for pure inference
        for s in tqdm(range(0, len(pair_users), chunk_pairs), desc="Model inference"):
            e = min(s + chunk_pairs, len(pair_users))

            u = torch.as_tensor(pair_users[s:e], device=device, dtype=torch.long)
            it = torch.as_tensor(pair_items[s:e], device=device, dtype=torch.long)

            # IMPORTANT: adapt unpack_input to accept tensors (recommended).
            # If your unpack_input expects an iterable of pairs, do:
            # test_data = unpack_input(opt, zip(u.tolist(), it.tolist()))
            test_data = unpack_input(opt, (u, it))  # <-- best if you can support this

            if opt.transnet:
                out, _ = model(test_data)
                out = out[1]
            else:
                out = model(test_data)

            all_scores[s:e] = out.detach().float().view(-1).cpu().numpy()

    # 3) Compute per-user ranking metrics
    recall_values, ndcg_values, map_values, mrr_values, diversity_values = [], [], [], [], []

    for (start, end, candidates, pos_items) in tqdm(users_meta, desc="Metrics"):
        scores = all_scores[start:end]

        # rank candidates by score DESC
        order = np.argsort(-scores)
        ranked_items = [candidates[i] for i in order]
        topk_items = ranked_items[:K]

        ndcg = calculate_ndcg_at_k(recommended_list=topk_items, ground_truth=pos_items, k=K)
        mp   = calculate_map_at_k (recommended_list=topk_items, ground_truth=pos_items, k=K)
        mrr  = calculate_mrr_at_k (recommended_list=topk_items, ground_truth=pos_items, k=K)
        rec  = calculate_recall_at_k(recommended_list=topk_items, ground_truth=pos_items, k=K)

        recall_values.append(rec)
        ndcg_values.append(ndcg)
        map_values.append(mp)
        mrr_values.append(mrr)

        # Diversity (optional): consider computing on top-K only (much cheaper)
        try:
            # If docs live inside test_data, you likely cannot access them here anymore.
            # Better: fetch embeddings by itemId from a precomputed matrix: item_doc_emb[item_id]
            # then compute intra-list diversity on topk_items.
            diversity_values.append(np.nan)  # placeholder
        except Exception:
            diversity_values.append(np.nan)

    return {
        "recall": recall_values,
        "ndcg": ndcg_values,
        "map": map_values,
        "mrr": mrr_values,
        "diversity": diversity_values,
    }
