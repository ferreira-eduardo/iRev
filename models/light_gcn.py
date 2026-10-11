# -*- coding: utf-8 -*-
"""
LightGCN adapted to iRev from the reference PyTorch implementation by Jianbai Ye
(gusye@mail.ustc.edu.cn) of:
    Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network
    for Recommendation. SIGIR 2020.

Kept as published: normalized user-item graph D^-1/2 A D^-1/2 built from the train
interactions, K propagation layers without weights/nonlinearities, layer-mean
readout, N(0, 0.1) init, and BPR loss + decay * L2 on the layer-0 (ego) embeddings,
optimized with plain Adam (main.py drops Adam's weight_decay for BPR models).

iRev integration:
    - objective = "bpr": main.py trains it with bpr_loss on (user, pos, sampled neg)
      triples instead of the rating MSE, and selects the epoch by validation nDCG@10.
    - forward() returns (score, score) for opt.direct_output, like LETTER.
    - user_side/item_side/score_matrix let main.py's factorized_scorer rank the
      full catalog with one matmul.
The graph's dropout option (off by default in the paper's experiments) is not ported.
"""

import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    objective = "bpr"

    def __init__(self, opt):
        super(LightGCN, self).__init__()
        self.opt = opt
        self.num_fea = 1  # direct_output: bypasses fusion/prediction layers

        self.num_users = opt.user_num
        self.num_items = opt.item_num
        self.n_layers = opt.lgcn_layers
        self.decay = opt.lgcn_decay

        self.embedding_user = nn.Embedding(self.num_users, opt.lgcn_dim)
        self.embedding_item = nn.Embedding(self.num_items, opt.lgcn_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        # Not a registered buffer, same as LETTER's rating matrix: it is rebuilt
        # from Train.npy and must not bloat every checkpoint.
        device = 'cuda' if opt.use_gpu else 'cpu'
        train = np.load(os.path.join(opt.data_root, "train", "Train.npy"))
        self.graph = self._normalized_graph(train, self.num_users, self.num_items).to(device)

    @staticmethod
    def _normalized_graph(train, num_users, num_items):
        """D^-1/2 A D^-1/2 for A = [[0, R], [R^T, 0]], R the binary train matrix."""
        pairs = np.unique(train[:, :2].astype(np.int64), axis=0)
        R = sp.csr_matrix((np.ones(len(pairs), dtype=np.float32), (pairs[:, 0], pairs[:, 1])),
                          shape=(num_users, num_items))
        A = sp.bmat([[None, R], [R.T, None]], format="csr")
        deg = np.asarray(A.sum(axis=1)).ravel()
        d_inv_sqrt = np.zeros_like(deg)
        d_inv_sqrt[deg > 0] = deg[deg > 0] ** -0.5
        A = sp.diags(d_inv_sqrt).dot(A).dot(sp.diags(d_inv_sqrt)).tocoo()
        indices = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
        values = torch.from_numpy(A.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, size=A.shape).coalesce()

    def computer(self):
        """Propagates the embeddings and returns the layer-mean (users, items)."""
        all_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embs.append(all_emb)
        light_out = torch.stack(embs, dim=1).mean(dim=1)
        return torch.split(light_out, [self.num_users, self.num_items])

    def bpr_loss(self, users, pos, neg):
        all_users, all_items = self.computer()
        users_emb, pos_emb, neg_emb = all_users[users], all_items[pos], all_items[neg]

        reg_loss = (1 / 2) * (self.embedding_user(users).norm(2).pow(2) +
                              self.embedding_item(pos).norm(2).pow(2) +
                              self.embedding_item(neg).norm(2).pow(2)) / float(len(users))
        pos_scores = (users_emb * pos_emb).sum(dim=1)
        neg_scores = (users_emb * neg_emb).sum(dim=1)
        loss = F.softplus(neg_scores - pos_scores).mean()
        return loss + self.decay * reg_loss

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
