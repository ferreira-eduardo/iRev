# -*- coding: utf-8 -*-
"""
RLMRec-Con on LightGCN, ported to iRev from the authors' PyTorch code
(HKUDS/RLMRec, encoder/models/general_cf/lightgcn_plus.py) of:
    Xubin Ren et al. Representation Learning with Large Language Models for
    Recommendation. WWW 2024.

Kept as released (LightGCN_plus + lightgcn_plus.yml):
    - LightGCN backbone of their SSLRec code: D^-1/2 A D^-1/2 without self-loops,
      SUM readout over layers, edge dropout (keep_rate, no rescaling) while training,
      Xavier-uniform embeddings, d = 32;
    - contrastive alignment: an MLP (profile_dim -> (profile_dim + d)/2 -> d, LeakyReLU)
      maps the LLM-profile embeddings to the CF space; InfoNCE (kd_temperature) between
      the batch users/positives/negatives and their mapped profiles, with all users'
      profiles as the user negatives and the batch's positives' profiles as the item ones;
    - loss = BPR (summed / batch) + reg_weight * sum ||W||^2 over ALL parameters
      + kd_weight * InfoNCE / batch; score = <u, i>.

Profiles: preprocess_data/generate_rlmrec_profiles.py (local Qwen3-14B instead of
gpt-3.5-turbo, train reviews only) + build_rlmrec_fixed_split.py (e5-base-v2 instead of
text-embedding-ada-002).

iRev integration (as LightGCN): objective "bpr" (main.py trains with bpr_loss on
(user, pos, sampled neg) and selects the epoch by validation nDCG@10, Adam without
weight_decay); user_side/item_side/score_matrix give factorized_scorer the full ranking.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .light_gcn import LightGCN


def _infonce(embeds1, embeds2, all_embeds2, temp):
    """RLMRec's cal_infonce_loss (summed over the rows)."""
    e1, e2, all2 = (F.normalize(x, dim=-1, eps=1e-8) for x in (embeds1, embeds2, all_embeds2))
    nume = -(e1 * e2).sum(-1) / temp
    deno = torch.logsumexp(e1 @ all2.t() / temp, dim=-1)
    return (nume + deno).sum()


class RLMRec(nn.Module):
    objective = "bpr"

    def __init__(self, opt):
        super(RLMRec, self).__init__()
        self.opt = opt
        self.num_fea = 1  # direct_output: bypasses fusion/prediction layers

        self.num_users = opt.user_num
        self.num_items = opt.item_num
        self.n_layers = opt.rlmrec_layers
        self.keep_rate = opt.rlmrec_keep_rate
        self.reg_weight = opt.rlmrec_reg_weight
        self.kd_weight = opt.rlmrec_kd_weight
        self.kd_temperature = opt.rlmrec_kd_temperature
        dim = opt.rlmrec_dim

        self.user_embeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_users, dim)))
        self.item_embeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_items, dim)))

        # Not registered buffers, as LightGCN's graph: rebuilt from the fold's files
        device = 'cuda' if opt.use_gpu else 'cpu'
        self.usrprf = torch.from_numpy(np.load(opt.rlmrec_user_prf_path).astype(np.float32)).to(device)
        self.itmprf = torch.from_numpy(np.load(opt.rlmrec_item_prf_path).astype(np.float32)).to(device)
        prf_dim = self.usrprf.shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(prf_dim, (prf_dim + dim) // 2),
            nn.LeakyReLU(),
            nn.Linear((prf_dim + dim) // 2, dim),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        train = np.load(os.path.join(opt.data_root, "train", "Train.npy"))
        # same D^-1/2 A D^-1/2 as RLMRec's _make_torch_adj(self_loop=False)
        self.graph = LightGCN._normalized_graph(train, self.num_users, self.num_items).to(device)

    def _edge_drop(self):
        """SpAdjEdgeDrop: keeps each edge with prob keep_rate, without rescaling."""
        if self.keep_rate >= 1.0:
            return self.graph
        idx, val = self.graph.indices(), self.graph.values()
        mask = torch.rand(val.shape, device=val.device) < self.keep_rate
        return torch.sparse_coo_tensor(idx[:, mask], val[mask], self.graph.shape)

    def computer(self):
        adj = self._edge_drop() if self.training else self.graph
        emb = torch.cat([self.user_embeds, self.item_embeds])
        embs = [emb]
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(adj, emb)
            embs.append(emb)
        return torch.split(sum(embs), [self.num_users, self.num_items])

    def bpr_loss(self, users, pos, neg):
        """Named bpr_loss for main.py's BPR path: BPR + reg + contrastive alignment."""
        user_embeds, item_embeds = self.computer()
        anc, p, n = user_embeds[users], item_embeds[pos], item_embeds[neg]
        bpr = F.softplus((anc * n).sum(-1) - (anc * p).sum(-1)).sum() / len(users)
        reg = self.reg_weight * sum(w.pow(2).sum() for w in self.parameters())

        usrprf, itmprf = self.mlp(self.usrprf), self.mlp(self.itmprf)
        ancprf, posprf, negprf = usrprf[users], itmprf[pos], itmprf[neg]
        kd = (_infonce(anc, ancprf, usrprf, self.kd_temperature) +
              _infonce(p, posprf, posprf, self.kd_temperature) +
              _infonce(n, negprf, negprf, self.kd_temperature)) / len(users)
        return bpr + reg + self.kd_weight * kd

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
