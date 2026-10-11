# -*- coding: utf-8 -*-
"""
LETTER expects, per fold (produced by preprocess_data/complement_folds.py):
    opt.letter_user_emb_path / opt.letter_item_emb_path: mean-pooled semantic
        review embedding per user/item, shaped (opt.user_num or opt.item_num, D)
    opt.letter_{user,item}_emb_{pos,neg}_path: same, restricted to reviews with
        rating above/at-or-below the pivot used at preprocessing time
    opt.letter_rating_matrix_path: sparse (opt.user_num, opt.item_num) train-fold
        rating matrix (scipy .npz), used to build the KNN similarity graph.
        Falls back to the dense opt.ratingMatrix_path .npy when the .npz is absent
        (the dense one is only written for small catalogs).
"""

import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class LETTER(nn.Module):
    def __init__(self, opt):
        super(LETTER, self).__init__()
        self.opt = opt
        self.num_fea = 1  # direct_output: bypasses fusion/prediction layers

        self.num_users = opt.user_num
        self.num_items = opt.item_num
        hidden_dim = opt.letter_hidden_dim

        def load_emb(path):
            weights = torch.from_numpy(np.load(path).astype(np.float32))
            return nn.Embedding.from_pretrained(weights, freeze=True)

        self.user_embedding = load_emb(opt.letter_user_emb_path)
        self.item_embedding = load_emb(opt.letter_item_emb_path)
        self.user_pos_embedding = load_emb(opt.letter_user_emb_pos_path)
        self.user_neg_embedding = load_emb(opt.letter_user_emb_neg_path)
        self.item_pos_embedding = load_emb(opt.letter_item_emb_pos_path)
        self.item_neg_embedding = load_emb(opt.letter_item_emb_neg_path)
        emb_dim = self.user_embedding.embedding_dim

        # Not a registered buffer: it must not bloat every saved checkpoint with a
        # full user x item matrix, so device placement is handled manually below.
        # Kept sparse and row-L2-normalized: a dense matrix is infeasible on large
        # catalogs (Musical Instruments: 214k x 123k ~ 105GB).
        device = 'cuda' if opt.use_gpu else 'cpu'
        if os.path.exists(opt.letter_rating_matrix_path):
            rating_matrix = sp.load_npz(opt.letter_rating_matrix_path)
        else:
            rating_matrix = sp.csr_matrix(np.load(opt.ratingMatrix_path))
        rating_matrix = rating_matrix.astype(np.float32)
        self.user_rating = self._normalized_sparse(rating_matrix, device)
        self.item_rating = self._normalized_sparse(rating_matrix.T, device)

        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        # Zero, not nn.Embedding's N(0, 1): ~71% of the catalog has a single train
        # interaction, so those biases stay near their init and random N(0, 1) noise
        # (larger than the target's spread) decides the full-catalog ranking.
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        def projection(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(opt.drop_out),
            )

        self.ruFC = projection(emb_dim)
        self.riFC = projection(emb_dim)
        self.rupFC = projection(emb_dim)
        self.runFC = projection(emb_dim)
        self.ripFC = projection(emb_dim)
        self.rinFC = projection(emb_dim)

        self.ugnn1 = nn.Linear(hidden_dim, hidden_dim)
        self.ignn1 = nn.Linear(hidden_dim, hidden_dim)
        self.upnn1 = nn.Linear(hidden_dim, hidden_dim)
        self.ipnn1 = nn.Linear(hidden_dim, hidden_dim)
        self.unnn1 = nn.Linear(hidden_dim, hidden_dim)
        self.innn1 = nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    def _normalized_sparse(matrix, device):
        """Row-L2-normalized scipy matrix -> torch sparse COO, so that a sparse x
        dense product of two rows is their cosine similarity."""
        matrix = sp.csr_matrix(matrix)
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        matrix = sp.diags(1.0 / np.maximum(norms, 1e-12)).dot(matrix).tocoo()
        indices = torch.from_numpy(np.vstack([matrix.row, matrix.col]).astype(np.int64))
        values = torch.from_numpy(matrix.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, size=matrix.shape).coalesce().to(device)

    def _topk_neighbors(self, profile_matrix, ids, top_k):
        device = ids.device
        total = profile_matrix.size(0)

        with torch.no_grad():
            query = profile_matrix.index_select(0, ids).to_dense()
            sim = torch.sparse.mm(profile_matrix, query.t()).t()
            final_values, final_indices = torch.topk(sim, k=min(top_k, total), dim=1)
        top_k = final_indices.size(1)

        sparse_values = F.softmax(final_values, dim=1).flatten()
        rows = torch.arange(ids.size(0), device=device).view(-1, 1).repeat(1, top_k).flatten()
        cols = final_indices.flatten()

        return torch.sparse_coo_tensor(
            torch.stack([rows, cols]), sparse_values,
            size=(ids.size(0), total), device=device,
        ).coalesce()

    @staticmethod
    def _sparse_projection(sim_matrix, embedding_layer, projection_layer, ids):
        neighbor_indices = sim_matrix.indices()[1].unique()
        all_needed = torch.unique(torch.cat([ids, neighbor_indices]))
        projected = projection_layer(embedding_layer.weight[all_needed])

        full = torch.zeros((embedding_layer.num_embeddings, projected.size(1)), device=ids.device)
        full[all_needed] = projected
        return full

    def user_side(self, uids):
        """(B, 2*hidden + 1): [user_r, user_pos_r - user_neg_r, user_bias].
        Each row depends only on its own uid, so users and items can be encoded
        separately and scored with a matmul (main.py's factorized_scorer)."""
        num_u = max(1, int(np.ceil(self.opt.letter_edge_ratio / 100.0 * self.num_users)))
        u_sims = self._topk_neighbors(self.user_rating, uids, num_u)

        u_emb_all = self._sparse_projection(u_sims, self.user_embedding, self.ruFC, uids)
        up_emb_all = self._sparse_projection(u_sims, self.user_pos_embedding, self.rupFC, uids)
        un_emb_all = self._sparse_projection(u_sims, self.user_neg_embedding, self.runFC, uids)

        user_r = self.ugnn1(torch.sparse.mm(u_sims, u_emb_all)) + u_emb_all[uids]
        user_pos_r = self.upnn1(torch.sparse.mm(u_sims, up_emb_all)) + up_emb_all[uids]
        user_neg_r = self.unnn1(torch.sparse.mm(u_sims, un_emb_all)) + un_emb_all[uids]

        return torch.cat([user_r, user_pos_r - user_neg_r, self.user_bias(uids)], dim=1)

    def item_side(self, iids):
        """(B, 2*hidden + 1): [item_r + item_pos_r - item_neg_r, item_r, item_bias]."""
        num_i = max(1, int(np.ceil(self.opt.letter_edge_ratio / 100.0 * self.num_items)))
        i_sims = self._topk_neighbors(self.item_rating, iids, num_i)

        i_emb_all = self._sparse_projection(i_sims, self.item_embedding, self.riFC, iids)
        ip_emb_all = self._sparse_projection(i_sims, self.item_pos_embedding, self.ripFC, iids)
        in_emb_all = self._sparse_projection(i_sims, self.item_neg_embedding, self.rinFC, iids)

        item_r = self.ignn1(torch.sparse.mm(i_sims, i_emb_all)) + i_emb_all[iids]
        item_pos_r = self.ipnn1(torch.sparse.mm(i_sims, ip_emb_all)) + ip_emb_all[iids]
        item_neg_r = self.innn1(torch.sparse.mm(i_sims, in_emb_all)) + in_emb_all[iids]

        return torch.cat([item_r + item_pos_r - item_neg_r, item_r, self.item_bias(iids)], dim=1)

    @staticmethod
    def score_matrix(user_side, item_side):
        """(U, I) scores; row-wise this is exactly forward()'s prediction:
        u.i + up.i - un.i + u.ip - u.in = [u, up - un] . [i + ip - in, i]."""
        return user_side[:, :-1] @ item_side[:, :-1].t() + user_side[:, -1:] + item_side[:, -1]

    def forward(self, datas):
        _, _, uids, iids, _, _, _, _ = datas
        user_side, item_side = self.user_side(uids), self.item_side(iids)
        prediction = (user_side[:, :-1] * item_side[:, :-1]).sum(1) + user_side[:, -1] + item_side[:, -1]
        return prediction, prediction
