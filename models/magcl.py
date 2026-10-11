# -*- coding: utf-8 -*-
"""
MAGCL ported to PyTorch/iRev from the authors' TensorFlow 1 code (model/MAGCL.py) of:
    Ke Wang et al. Multi-aspect Graph Contrastive Learning for Review-enhanced
    Recommendation. ACM TOIS.

Kept as released:
    - graph 1: user-item interactions + self-loops, row-normalized D^-1 (A + I);
      graph 2: 0.1 * (user-user, item-item review-similarity graphs) + I, row-normalized
      (similarity graphs from preprocess_data/build_magcl_fixed_split.py);
    - score = <u, i> ("MF" mode) with u = LightGCN(G1, E1) + 0.1 * sum_k LightGCN(G2, E2k),
      k = 1..5 independent embedding tables on the same review graph ("multi-aspect");
    - contrastive view pair: a weighted GCN with LeakyReLU on G1 (from E1) vs
      sum_k LightGCN(G2, E2k), InfoNCE on cosines divided by 5;
    - loss: sigmoid cross-entropy over 1 positive (weight 5) + 4 sampled negatives,
      plus ssl_weight * InfoNCE; layer-mean readout over K = 3 layers, d = 64.

Changed, and why:
    - init N(0, 0.01) truncated at 2 std: the released tf.truncated_normal_initializer(0.01)
      passes 0.01 as the MEAN (std 1.0), clearly not the intent;
    - L2 (magcl_decay) on the batch's layer-0 embeddings, as LightGCN here: the released
      regularizers are declared but never added to the loss;
    - InfoNCE averaged (not summed) over the batch's unique users/items, so its weight
      does not scale with the batch size (released: batch of 100 rows, duplicates kept);
    - no lr schedule: the released one edits args.lr after the optimizer is built,
      so it never takes effect.

iRev integration (as LightGCN): objective "bpr" (main.py calls bpr_loss with
num_negatives sampled items per positive and selects the epoch by validation nDCG@10);
user_side/item_side/score_matrix give main.py's factorized_scorer the full-catalog ranking.
"""

import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ASPECT_VIEWS = 5
ASPECT_WEIGHT = 0.1      # weight of each review-graph view in the final embedding
REVIEW_EDGE_WEIGHT = 0.1  # weight of the similarity edges in graph 2
POS_WEIGHT = 5.0
SSL_TEMPERATURE = 5.0


def _sparse_tensor(A, device):
    A = A.tocoo()
    indices = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
    values = torch.from_numpy(A.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, size=A.shape).coalesce().to(device)


def _row_normalized(A):
    """D^-1 (A + I), MAGCL's normalized_adj_single."""
    A = (A + sp.eye(A.shape[0], dtype=np.float32)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    return sp.diags(1.0 / deg).dot(A)


class MAGCL(nn.Module):
    objective = "bpr"
    num_negatives = 4

    def __init__(self, opt):
        super(MAGCL, self).__init__()
        self.opt = opt
        self.num_fea = 1  # direct_output: bypasses fusion/prediction layers

        self.num_users = opt.user_num
        self.num_items = opt.item_num
        self.n_layers = opt.magcl_layers
        self.decay = opt.magcl_decay
        self.ssl_weight = opt.magcl_ssl_weight
        dim = opt.magcl_dim

        def table(n):
            emb = nn.Embedding(n, dim)
            nn.init.trunc_normal_(emb.weight, std=0.01, a=-0.02, b=0.02)
            return emb

        self.user_emb = nn.ModuleList([table(self.num_users) for _ in range(NUM_ASPECT_VIEWS + 1)])
        self.item_emb = nn.ModuleList([table(self.num_items) for _ in range(NUM_ASPECT_VIEWS + 1)])
        self.ssl_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n_layers)])
        for layer in self.ssl_layers:
            nn.init.trunc_normal_(layer.weight, std=0.01, a=-0.02, b=0.02)
            nn.init.zeros_(layer.bias)

        # Not registered buffers, as LightGCN's graph: rebuilt from the fold's files
        device = 'cuda' if opt.use_gpu else 'cpu'
        train = np.load(os.path.join(opt.data_root, "train", "Train.npy"))
        pairs = np.unique(train[:, :2].astype(np.int64), axis=0)
        R = sp.csr_matrix((np.ones(len(pairs), dtype=np.float32), (pairs[:, 0], pairs[:, 1])),
                          shape=(self.num_users, self.num_items))
        self.graph1 = _sparse_tensor(_row_normalized(sp.bmat([[None, R], [R.T, None]])), device)
        U, I = sp.load_npz(opt.magcl_user_graph_path), sp.load_npz(opt.magcl_item_graph_path)
        self.graph2 = _sparse_tensor(_row_normalized(REVIEW_EDGE_WEIGHT * sp.block_diag([U, I])), device)

    def _ego(self, k):
        return torch.cat([self.user_emb[k].weight, self.item_emb[k].weight])

    def _lightgcn(self, graph, k):
        emb = self._ego(k)
        embs = [emb]
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(graph, emb)
            embs.append(emb)
        return torch.stack(embs, dim=1).mean(dim=1)

    def _gcn_ssl(self):
        emb = self._ego(0)
        embs = [emb]
        for layer in self.ssl_layers:
            emb = F.leaky_relu(layer(torch.sparse.mm(self.graph1, emb)), 0.2)
            embs.append(emb)
        return torch.stack(embs, dim=1).mean(dim=1)

    def _split(self, emb):
        return torch.split(emb, [self.num_users, self.num_items])

    def computer(self, with_views=False):
        """(users, items) final embeddings; with_views also returns the two SSL views."""
        aspects = [self._lightgcn(self.graph2, k) for k in range(1, NUM_ASPECT_VIEWS + 1)]
        final = self._lightgcn(self.graph1, 0) + ASPECT_WEIGHT * sum(aspects)
        if not with_views:
            return self._split(final)
        return self._split(final), self._split(self._gcn_ssl()), self._split(sum(aspects))

    def bpr_loss(self, users, pos, neg):
        """users, pos: (B,); neg: (B, num_negatives). Named bpr_loss for main.py's BPR path."""
        (all_u, all_i), (v1_u, v1_i), (v2_u, v2_i) = self.computer(with_views=True)
        items = torch.cat([pos.unsqueeze(1), neg], dim=1)  # (B, 1 + n_neg)
        logits = (all_u[users].unsqueeze(1) * all_i[items]).sum(-1)
        labels = torch.zeros_like(logits)
        labels[:, 0] = 1.0
        weights = 1.0 + (POS_WEIGHT - 1.0) * labels
        rec_loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)

        u, i = torch.unique(users), torch.unique(items)
        z1 = F.normalize(torch.cat([v1_i[i], v1_u[u]]), dim=1)
        z2 = F.normalize(torch.cat([v2_i[i], v2_u[u]]), dim=1)
        sim = z1 @ z2.t() / SSL_TEMPERATURE
        ssl_loss = (torch.logsumexp(sim, dim=1) - sim.diagonal()).mean()

        reg = sum(self.user_emb[k](u).pow(2).sum() + self.item_emb[k](i).pow(2).sum()
                  for k in range(NUM_ASPECT_VIEWS + 1)) / (2 * len(users))
        return rec_loss + self.ssl_weight * ssl_loss + self.decay * reg

    def user_side(self, uids):
        return self.computer()[0][uids]

    def item_side(self, iids):
        return self.computer()[1][iids]

    @staticmethod
    def score_matrix(user_side, item_side):
        return user_side @ item_side.t()

    def forward(self, datas):
        _, _, uids, iids, _, _, _, _ = datas
        all_users, all_items = self.computer()
        prediction = (all_users[uids] * all_items[iids]).sum(dim=1)
        return prediction, prediction
