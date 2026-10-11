# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class PredictionLayer(nn.Module):
    '''
        RATING PREDICTION METHODS
        - LFM: Latent Factor Model
        - (N)FM: (Neural) Factorization Machine
        - MLP
        - SUM
    '''
    def __init__(self, opt):
        super(PredictionLayer, self).__init__()
        self.output = opt.output
        if opt.output == "fm":
            self.model = FM(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == "lfm":
            train_mean = float(np.load(f"{opt.data_root}/train/Train_Score.npy").mean())
            self.model = LFM(opt.feature_dim, opt.user_num, opt.item_num, train_mean)
        elif opt.output == 'mlp':
            self.model = MLP(opt.feature_dim)
        elif opt.output == 'nfm':
            self.model = NFM(opt.feature_dim)
        else:
            self.model = torch.sum

    def forward(self, feature, uid, iid):
        if self.output == "lfm" or "fm" or "nfm":
            return self.model(feature, uid, iid)
        else:
            return self.model(feature, 1, keepdim=True)


class LFM(nn.Module):
    '''
        LATENT FACTOR MODEL
    '''

    def __init__(self, dim, user_num, item_num, train_mean=0.0):
        super(LFM, self).__init__()

        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        # -------------------------LFM-user/item-bias-----------------------
        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))
        self.global_bias = nn.Parameter(torch.tensor(train_mean, dtype=torch.float32))

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc.bias, a=0.5, b=1.5)
        nn.init.uniform_(self.b_users, a=0.5, b=1.5)
        nn.init.uniform_(self.b_items, a=0.5, b=1.5)
        # Start at the train mean instead of the ~3 the uniform draws above add up to: on
        # IMDB (1-10, mean 7.4) the small learning rates could not close that gap in time.
        # The draws stay (then zeroed) so the RNG stream, and with it batch order, is unchanged.
        with torch.no_grad():
            self.fc.bias.zero_()
            self.b_users.zero_()
            self.b_items.zero_()

    def rescale_sigmoid(self, score, a, b):
        return a + torch.sigmoid(score) * (b - a)

    def forward(self, feature, user_id, item_id):
        # return self.rescale_sigmoid(self.fc(feature), 1.0, 5.0) + self.b_users[user_id] + self.b_items[item_id]
        return self.fc(feature) + self.b_users[user_id] + self.b_items[item_id] + self.global_bias


class NFM(nn.Module):
    '''
        NEURAL FACTORIZATION MACHINE
    '''
    def __init__(self, dim):
        super(NFM, self).__init__()
        self.dim = dim
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        # ------------------------------FM----------------------------------
        self.fm_V = nn.Parameter(torch.randn(16, dim))
        self.mlp = nn.Linear(16, 16)
        self.h = nn.Linear(16, 1, bias=False)
        self.drop_out = nn.Dropout(0.5)
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0.1)
        nn.init.uniform_(self.fm_V, -0.1, 0.1)
        nn.init.uniform_(self.h.weight, -0.1, 0.1)

    def forward(self, input_vec, *args):
        fm_linear_part = self.fc(input_vec)
        fm_interactions_1 = torch.mm(input_vec, self.fm_V.t())
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2), torch.pow(self.fm_V, 2).t())
        bilinear = 0.5 * (fm_interactions_1 - fm_interactions_2)

        out = F.relu(self.mlp(bilinear))
        out = self.drop_out(out)
        out = self.h(out) + fm_linear_part
        return out


class FM(nn.Module):
    '''
        FACTORIZATION MACHINE
    '''

    def __init__(self, dim, user_num, item_num):
        super(FM, self).__init__()
        self.dim = dim
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        # ------------------------------FM----------------------------------
        self.fm_V = nn.Parameter(torch.randn(dim, 10))
        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)
        nn.init.uniform_(self.fm_V, -0.05, 0.05)

    def build_fm(self, input_vec):
        '''
        y = w_0 + \sum {w_ix_i} + \sum_{i=1}\sum_{j=i+1}<v_i, v_j>x_ix_j
        factorization machine layer
        refer: https://github.com/vanzytay/KDD2018_MPCN/blob/master/tylib/lib
                      /compose_op.py#L13
        '''
        # linear part: first two items
        fm_linear_part = self.fc(input_vec)

        fm_interactions_1 = torch.mm(input_vec, self.fm_V)
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2),
                                     torch.pow(self.fm_V, 2))
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2, 1, keepdim=True) + fm_linear_part
        return fm_output

    def forward(self, feature, uids, iids):
        fm_out = self.build_fm(feature)
        return fm_out + self.b_users[uids] + self.b_items[iids]


class MLP(nn.Module):
    '''
        MULTILAYER PERCEPTRON
    '''

    def __init__(self, dim):
        super(MLP, self).__init__()
        self.dim = dim
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, 0.1, 0.1)
        nn.init.uniform_(self.fc.bias, a=0, b=0.2)

    def forward(self, feature, *args, **kwargs):
        return F.relu(self.fc(feature))
