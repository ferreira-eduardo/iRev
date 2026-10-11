# -*- coding: utf-8 -*-
"""
KANN (Knowledge-graph Attentive Neural Network), ported from
add_eduardo/KANN/kann.py (KA class) into iRev's opt/datas convention -- see
models/letter.py for the same porting pattern.

Expects, per fold (produced by add_eduardo/generate_dataset_kann.py):
    opt.kann_re_emb_path: entity embeddings shaped (num_entities, 3 * opt.kann_d_model),
        the concatenated (head, relation, tail) TransE-style embedding per review entity,
        mean-pooled over every triple the entity appears in.
    opt.kann_ori_emb_path: entity embeddings shaped (num_entities, opt.kann_d_model),
        the plain (non-relation-aware) entity embedding.
    opt.kann_user_entities_path / opt.kann_item_entities_path: padded sequences of
        entity ids per user/item, shaped (opt.user_num, Su) / (opt.item_num, Si) --
        index into the two embedding tables above. Padding id is 0.

Loaded directly here (not through the generic `datas` tuple) because the shared
opt.users_review_list / opt.items_review_list are word-token review arrays used by
every other model -- KANN needs its own entity-id sequences instead, same reasoning
as opt.ratingMatrix_path in models/letter.py.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Interacter(nn.Module):
    """Stack of self-/cross-attention Transformer blocks (pre-LN residual style)."""

    def __init__(self, num_layers, d_model, num_heads, dff, rate):
        super().__init__()
        self.num_layers = num_layers
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=rate, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model, eps=1e-6) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model, eps=1e-6) for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, dff), nn.ReLU(), nn.Linear(dff, d_model))
            for _ in range(num_layers)
        ])

    def forward(self, q, k, v, key_padding_mask=None):
        x = q
        for l in range(self.num_layers):
            attn_out, _ = self.attn[l](x, k, v, key_padding_mask=key_padding_mask, need_weights=False)
            x = self.norm1[l](x + self.dropout(attn_out))
            x = self.norm2[l](x + self.dropout(self.ffn[l](x)))
        return x


class KANN(nn.Module):
    def __init__(self, opt):
        super(KANN, self).__init__()
        self.opt = opt
        self.num_fea = 1  # direct_output: bypasses fusion/prediction layers

        self.num_users = opt.user_num
        self.num_items = opt.item_num
        d_model = opt.kann_d_model
        self.attn_dim = 4 * d_model
        rate = opt.drop_out

        # Score = global mean + user/item bias + residual from the network. The original
        # KANN starts every bias at 0.1, so the network itself has to produce the whole
        # rating level (~4.2) through its ReLU/product chain; that large gain amplifies the
        # small train/eval activation shift caused by dropout into a ~+3 star offset in
        # eval mode. Starting global_bias at the train mean keeps the network's output small.
        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        train_mean = float(np.load(f"{opt.data_root}/train/Train_Score.npy").mean())
        self.global_bias = nn.Parameter(torch.tensor(train_mean, dtype=torch.float32))

        re_weight = torch.from_numpy(np.load(opt.kann_re_emb_path).astype(np.float32))
        ori_weight = torch.from_numpy(np.load(opt.kann_ori_emb_path).astype(np.float32))
        self.re_emb = nn.Embedding.from_pretrained(re_weight, freeze=opt.kann_freeze_entity_emb)
        self.ori_emb = nn.Embedding.from_pretrained(ori_weight, freeze=opt.kann_freeze_entity_emb)

        # Not registered as buffers/parameters: mirrors opt.user_rating/opt.item_rating
        # in models/letter.py -- these must not bloat every saved checkpoint with the
        # full user/item x entity-sequence table, so device placement is manual below.
        device = 'cuda' if opt.use_gpu else 'cpu'
        self.user_entities = torch.from_numpy(
            np.load(opt.kann_user_entities_path).astype(np.int64)).to(device)
        self.item_entities = torch.from_numpy(
            np.load(opt.kann_item_entities_path).astype(np.int64)).to(device)

        self.layernorm_conq = nn.LayerNorm(self.attn_dim, eps=1e-6)
        self.layernorm_conk = nn.LayerNorm(self.attn_dim, eps=1e-6)
        self.dropout = nn.Dropout(rate)

        self.attention_u = _Interacter(opt.kann_num_layers, self.attn_dim, opt.kann_num_heads, opt.kann_dff, rate)
        self.attention_i = _Interacter(opt.kann_num_layers, self.attn_dim, opt.kann_num_heads, opt.kann_dff, rate)
        self.interacter_u = _Interacter(opt.kann_num_layers, self.attn_dim, opt.kann_num_heads, opt.kann_dff, rate)
        self.interacter_i = _Interacter(opt.kann_num_layers, self.attn_dim, opt.kann_num_heads, opt.kann_dff, rate)

        self.layernorm_con_oriu = nn.LayerNorm(5 * d_model, eps=1e-6)
        self.layernorm_con_orii = nn.LayerNorm(5 * d_model, eps=1e-6)

        # After interaction: (B, 5*d) -> (B, d)
        self.after_inter_layer = nn.Linear(5 * d_model, d_model)

        # Act on the review-length axis after transpose: (B, 5*d, S) -> (B, 5*d, 1).
        # LazyLinear infers S (padded review-entity count) on the first forward call.
        self.prediction_u_layer = nn.LazyLinear(1)
        self.prediction_i_layer = nn.LazyLinear(1)

        self.prediction_layer = nn.Linear(d_model, 1)

    @staticmethod
    def _key_padding_mask(seq):
        # padding entity id is 0 (see add_eduardo/KANN/build_dataset.py:pad_reviews)
        return seq == 0

    def forward(self, datas):
        _, _, uids, iids, _, _, _, _ = datas
        u_batch = self.user_entities[uids]  # (B, Su)
        i_batch = self.item_entities[iids]  # (B, Si)

        u_mask = self._key_padding_mask(u_batch)
        i_mask = self._key_padding_mask(i_batch)

        q_context = self.re_emb(u_batch)  # (B, Su, 3d)
        k_context = self.re_emb(i_batch)  # (B, Si, 3d)
        q_ori = self.ori_emb(u_batch)     # (B, Su, d)
        k_ori = self.ori_emb(i_batch)     # (B, Si, d)

        ureview_emb = self.layernorm_conq(torch.cat([q_context, q_ori], dim=2))  # (B, Su, 4d)
        ireview_emb = self.layernorm_conk(torch.cat([k_context, k_ori], dim=2))  # (B, Si, 4d)

        scale = math.sqrt(self.attn_dim)
        ureview_emb = self.dropout(ureview_emb * scale)
        ireview_emb = self.dropout(ireview_emb * scale)

        # self-attn
        u_out = self.attention_u(ureview_emb, ureview_emb, ureview_emb, u_mask)
        i_out = self.attention_i(ireview_emb, ireview_emb, ireview_emb, i_mask)

        # cross-attn
        inter_out_u = self.interacter_u(u_out, i_out, i_out, i_mask)
        inter_out_i = self.interacter_i(i_out, u_out, u_out, u_mask)

        # concat original features again => 5d
        inter_out_u = self.layernorm_con_oriu(torch.cat([inter_out_u, q_ori], dim=2))  # (B, Su, 5d)
        inter_out_i = self.layernorm_con_orii(torch.cat([inter_out_i, k_ori], dim=2))  # (B, Si, 5d)

        inter_out_u = inter_out_u.transpose(1, 2)  # (B, 5d, Su)
        inter_out_i = inter_out_i.transpose(1, 2)  # (B, 5d, Si)

        out_u = F.relu(self.prediction_u_layer(inter_out_u)).squeeze(-1)  # (B, 5d)
        out_i = F.relu(self.prediction_i_layer(inter_out_i)).squeeze(-1)  # (B, 5d)

        interaction = F.relu(self.after_inter_layer(out_u * out_i))  # (B, d_model)
        # no ReLU here (unlike the original): the residual around the global mean must be
        # able to go negative
        scores = self.prediction_layer(interaction).squeeze(-1)  # (B,)

        scores = scores + self.user_bias(uids).squeeze(-1) + self.item_bias(iids).squeeze(-1) + self.global_bias
        return scores, scores
