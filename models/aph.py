# -*- coding: utf-8 -*-
"""
APH (Aspect Performance-aware Hypergraph Neural Network, Liu et al., WSDM'25), ported from
the authors' src/model.py + src/layer.py (APM, GAT2, GraphPool, FM_Layer).

Follows the paper rather than the released model.py, whose conv_pool() discards the
hypergraph output (the FM only ever saw id/word embedding rows), and whose attention
softmax is commented out:
    x_q    = MLP(x_s, x_o)                                   query from sentiment + other side
    pi_e   = LeakyReLU[(x_c W1) . (x_q W2 + x_a W3)]         per hyperedge e = (c, o, a, s)
    w_e    = softmax of pi over all hyperedges of the center c
    m_c    = ReLU(Lm(W4 sum_e w_e x_a))                      aspect-performance aggregation
    g_c    = ReLU(Lg(max over the top-t edges by pi of sigmoid(pi_e) x_a W4))   aspect fusion
    y_c    = m_c (+) g_c
    rating = FM(y_u (+) y_i) + b_u + b_i + global            second-order FM, d' = 4d
Users and items are encoded the same way: a user's hyperedges carry (aspect, sentiment,
item), an item's carry (aspect, sentiment, user). One hop, as in the paper.

Expects, per fold (produced by preprocess_data/build_aph_fixed_split.py):
    opt.aph_user_edges_path / opt.aph_item_edges_path: (user_num / item_num, E, 3) int,
        [aspect, sentiment, other id] per hyperedge, aspect 0 = padding.
    opt.aph_stats_path: json with num_aspects.

The FM over the concatenation [y_u, y_i] splits exactly into user-only terms, item-only
terms and an inner product, so user_side/item_side/score_matrix give main.py's
factorized_scorer the full-catalog ranking with one matmul.
"""

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

NUM_SENTIMENTS = 3  # negative / neutral / positive (build_aph_fixed_split.SENTIMENT_ID)


class AspectAggregator(nn.Module):
    """Aspect performance-aware hypergraph aggregation + aspect fusion for one side."""

    def __init__(self, dim, top_ratio, negative_slope=0.2):
        super().__init__()
        self.top_ratio = top_ratio
        self.negative_slope = negative_slope
        self.query = nn.Linear(2 * dim, dim)  # x_q = MLP(x_s, x_o)
        self.W = nn.ParameterList([nn.Parameter(torch.empty(dim, dim)) for _ in range(4)])
        for w in self.W:
            init.xavier_uniform_(w, 0.1)
        self.m_layer = nn.Linear(dim, dim)
        self.g_layer = nn.Linear(dim, dim)

    def forward(self, x_c, x_a, x_s, x_o, mask):
        # x_c (B, d); x_a, x_s, x_o (B, E, d); mask (B, E)
        W1, W2, W3, W4 = self.W
        x_q = self.query(torch.cat([x_s, x_o], dim=-1))
        pi = ((x_c @ W1).unsqueeze(1) * (x_q @ W2 + x_a @ W3)).sum(-1)
        # finite fill: a node without hyperedges would make an all -inf softmax, whose
        # backward is NaN even when its output is masked away
        pi = F.leaky_relu(pi, self.negative_slope).masked_fill(~mask, -1e9)

        has_edges = mask.any(1, keepdim=True)
        w = torch.softmax(pi, dim=1).masked_fill(~mask, 0.0)
        aspect_fea = x_a @ W4  # (B, E, d)
        m = F.relu(self.m_layer((w.unsqueeze(-1) * aspect_fea).sum(1)))

        # top-t edges by pi, t = ceil(top_ratio * degree) (GraphPool's topk ratio)
        t = torch.ceil(self.top_ratio * mask.sum(1, keepdim=True)).long()
        rank = pi.argsort(dim=1, descending=True).argsort(dim=1)
        top = (rank < t) & mask
        gated = aspect_fea * torch.sigmoid(pi.masked_fill(~mask, 0.0)).unsqueeze(-1)
        g = gated.masked_fill(~top.unsqueeze(-1), float("-inf")).max(1).values
        g = F.relu(self.g_layer(torch.where(has_edges, g, torch.zeros_like(g))))
        return torch.cat([m, g], dim=-1)  # (B, 2d)


class APH(nn.Module):
    def __init__(self, opt):
        super(APH, self).__init__()
        self.opt = opt
        self.num_fea = 1  # direct_output: bypasses fusion/prediction layers

        dim = opt.aph_dim
        with open(opt.aph_stats_path) as f:
            num_aspects = json.load(f)["num_aspects"]

        self.user_emb = nn.Embedding(opt.user_num, dim)
        self.item_emb = nn.Embedding(opt.item_num, dim)
        self.aspect_emb = nn.Embedding(num_aspects + 1, dim, padding_idx=0)
        self.sentiment_emb = nn.Embedding(NUM_SENTIMENTS, dim)
        for emb in (self.user_emb, self.item_emb, self.aspect_emb, self.sentiment_emb):
            init.xavier_uniform_(emb.weight, 0.1)

        self.user_agg = AspectAggregator(dim, opt.aph_top_ratio)
        self.item_agg = AspectAggregator(dim, opt.aph_top_ratio)

        # Not registered as buffers: same reasoning as models/kann.py's entity tables
        device = 'cuda' if opt.use_gpu else 'cpu'
        self.user_edges = torch.from_numpy(np.load(opt.aph_user_edges_path).astype(np.int64)).to(device)
        self.item_edges = torch.from_numpy(np.load(opt.aph_item_edges_path).astype(np.int64)).to(device)

        # FM_Layer over z = [y_u, y_i] (4d): linear w, factors V (4d x 4d), user/item biases.
        # Global bias at the train mean (the authors fix it at 3), as in models/kann.py.
        fm_dim = 4 * dim
        self.fm_linear = nn.Linear(fm_dim, 1, bias=False)
        self.fm_V = nn.Parameter(torch.empty(fm_dim, fm_dim))
        init.xavier_uniform_(self.fm_V, 0.1)
        self.user_bias = nn.Parameter(torch.zeros(opt.user_num))
        self.item_bias = nn.Parameter(torch.zeros(opt.item_num))
        train_mean = float(np.load(f"{opt.data_root}/train/Train_Score.npy").mean())
        self.global_bias = nn.Parameter(torch.tensor(train_mean, dtype=torch.float32))

    def no_decay_parameters(self):
        """Exempt from main.py's weight_decay (see the optimizer there)."""
        return [self.user_bias, self.item_bias, self.global_bias]

    def _encode(self, ids, center_emb, other_emb, edges_table, agg, exclude=None):
        edges = edges_table[ids]  # (B, E, 3)
        aspects, sentiments, others = edges.unbind(-1)
        mask = aspects > 0
        if exclude is not None:
            # train pairs: drop the hyperedges of the review being predicted, whose sentiment
            # gives the target away; val/test pairs never have them (train reviews only)
            mask = mask & (others != exclude.unsqueeze(1))
        return agg(center_emb(ids), self.aspect_emb(aspects), self.sentiment_emb(sentiments),
                   other_emb(others), mask)

    def _fm_half(self, y, rows, bias):
        """(B, 4d + 1): [y V_half, terms of the FM that depend on this side only]."""
        V = self.fm_V[rows]
        p = y @ V
        own = (y @ self.fm_linear.weight[0, rows] + 0.5 * (p.pow(2).sum(1) - (y.pow(2) @ V.pow(2)).sum(1)) + bias)
        return torch.cat([p, own.unsqueeze(1)], dim=1)

    def user_side(self, uids, exclude_iids=None):
        y_u = self._encode(uids, self.user_emb, self.item_emb, self.user_edges, self.user_agg, exclude_iids)
        half = y_u.size(1)
        return self._fm_half(y_u, slice(0, half), self.user_bias[uids])

    def item_side(self, iids, exclude_uids=None):
        y_i = self._encode(iids, self.item_emb, self.user_emb, self.item_edges, self.item_agg, exclude_uids)
        half = y_i.size(1)
        return self._fm_half(y_i, slice(half, 2 * half), self.item_bias[iids] + self.global_bias)

    @staticmethod
    def score_matrix(user_side, item_side):
        """(U, I) scores; row-wise this is exactly forward()'s FM prediction:
        0.5(|p_u + p_i|^2 - ...) = own_u + own_i + p_u . p_i."""
        return user_side[:, :-1] @ item_side[:, :-1].t() + user_side[:, -1:] + item_side[:, -1]

    def forward(self, datas):
        _, _, uids, iids, _, _, _, _ = datas
        if self.training and self.opt.aph_exclude_target:
            user_side, item_side = self.user_side(uids, iids), self.item_side(iids, uids)
        else:
            user_side, item_side = self.user_side(uids), self.item_side(iids)
        prediction = (user_side[:, :-1] * item_side[:, :-1]).sum(1) + user_side[:, -1] + item_side[:, -1]
        return prediction, prediction
