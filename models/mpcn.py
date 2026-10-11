# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MPCN(nn.Module):
    
    def __init__(self, opt, head=3):
        
        super(MPCN, self).__init__()

        self.opt = opt
        self.num_fea = 1 
        self.head = head
        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  
        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  

        self.fc_g1 = nn.Linear(opt.word_dim, opt.word_dim)
        self.fc_g2 = nn.Linear(opt.word_dim, opt.word_dim)

        self.vectorized_coatt_review = Co_Attention_Vec(opt.word_dim, head, gumbel=True, pooling='max')
        self.vectorized_coatt_word = Co_Attention_Vec(opt.word_dim, head, gumbel=False, pooling='avg')

        self.u_fc = self.fc_layer()
        self.i_fc = self.fc_layer()

        self.drop_out = nn.Dropout(opt.drop_out)
        self.reset_para()

    def fc_layer(self):
        return nn.Sequential(
            nn.Linear(self.opt.word_dim * self.head, self.opt.word_dim),
            nn.ReLU(),
            nn.Linear(self.opt.word_dim, self.opt.id_emb_size)
        )
    #forward otimizado
    def forward(self, datas):
        user_reviews, item_reviews, _, _, _, _, _, _ = datas

        # OTIMIZAÇÃO: Use apenas os primeiros X reviews/palavras se necessário
        # user_reviews = user_reviews[:, :10, :100]

        u_word_embs = self.user_word_embs(user_reviews)  # [B, R, L, D]
        i_word_embs = self.item_word_embs(item_reviews)

        # 1. Review Gate (Representação por Review)
        # u_reviews: [B, R, D]
        u_reviews = self.review_gate(u_word_embs)
        i_reviews = self.review_gate(i_word_embs)

        # --- CO-ATTENTION DE REVIEWS ---
        # Aqui L = R (Número de reviews, ex: 10 ou 20) -> Matriz pequena [B, H, 10, 10]
        u_reviews_exp = u_reviews.unsqueeze(1).expand(-1, self.head, -1, -1)
        i_reviews_exp = i_reviews.unsqueeze(1).expand(-1, self.head, -1, -1)
        p_u_r, p_i_r = self.vectorized_coatt_review(u_reviews_exp, i_reviews_exp)

        # Seleção das palavras baseada na review selecionada
        # u_words: [B, H, L, D] -> Aqui L é a quantidade de palavras de UMA review
        u_words = torch.einsum('bhr,brld->bhld', p_u_r.squeeze(-1), u_word_embs)
        i_words = torch.einsum('bhr,brld->bhld', p_i_r.squeeze(-1), i_word_embs)

        # --- CO-ATTENTION DE PALAVRAS ---
        # Aqui L = Word_Len (ex: 500) -> Matriz [B, H, 500, 500] -> ~1GB VRAM
        p_u_w, p_i_w = self.vectorized_coatt_word(u_words, i_words)

        # Feature Final
        u_w_fea = torch.einsum('bhld,bhl->bhd', u_words, p_u_w.squeeze(-1))  # [B, H, D]
        i_w_fea = torch.einsum('bhld,bhl->bhd', i_words, p_i_w.squeeze(-1))  # [B, H, D]

        u_fea = u_w_fea.reshape(user_reviews.size(0), -1)  # [B, H*D]
        i_fea = i_w_fea.reshape(item_reviews.size(0), -1)  # [B, H*D]

        u_fea = self.drop_out(self.u_fc(u_fea))
        i_fea = self.drop_out(self.i_fc(i_fea))

        return torch.stack([u_fea], 1), torch.stack([i_fea], 1)

    # def forward(self, datas):
    #
    #     user_reviews, item_reviews, _, _, _, _, _, _ = datas
    #     u_word_embs = self.user_word_embs(user_reviews)
    #     i_word_embs = self.item_word_embs(item_reviews)
    #
    #     u_reviews = self.review_gate(u_word_embs)
    #     i_reviews = self.review_gate(i_word_embs)
    #
    #     u_fea = []
    #     i_fea = []
    #     for i in range(self.head):
    #         r_coatt = self.review_coatt[i]
    #         w_coatt = self.word_coatt[i]
    #
    #         p_u, p_i = r_coatt(u_reviews, i_reviews)
    #         u_r_words = user_reviews.permute(0, 2, 1).float().bmm(p_u)
    #         i_r_words = item_reviews.permute(0, 2, 1).float().bmm(p_i)
    #         u_words = self.user_word_embs(u_r_words.squeeze(2).long())
    #         i_words = self.item_word_embs(i_r_words.squeeze(2).long())
    #         p_u, p_i = w_coatt(u_words, i_words)
    #         u_w_fea = u_words.permute(0, 2, 1).bmm(p_u).squeeze(2)
    #         i_w_fea = u_words.permute(0, 2, 1).bmm(p_i).squeeze(2)
    #         u_fea.append(u_w_fea)
    #         i_fea.append(i_w_fea)
    #
    #     u_fea = torch.cat(u_fea, 1)
    #     i_fea = torch.cat(i_fea, 1)
    #
    #     u_fea = self.drop_out(self.u_fc(u_fea))
    #     i_fea = self.drop_out(self.i_fc(i_fea))
    #
    #     return torch.stack([u_fea], 1), torch.stack([i_fea], 1)

    def review_gate(self, reviews):
        reviews = reviews.sum(2)
        return torch.sigmoid(self.fc_g1(reviews)) * torch.tanh(self.fc_g2(reviews))

    def reset_para(self):
        for fc in [self.fc_g1, self.fc_g2, self.u_fc[0], self.u_fc[-1], self.i_fc[0], self.i_fc[-1]]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.uniform_(fc.bias, -0.1, 0.1)

        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.user_word_embs.weight.data.copy_(w2v.cuda())
                self.item_word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.user_word_embs.weight.data.copy_(w2v)
                self.item_word_embs.weight.data.copy_(w2v)
        else:
            nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
            nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)


class Co_Attention_Vec(nn.Module):
    def __init__(self, dim, head, gumbel, pooling):
        super(Co_Attention_Vec, self).__init__()
        self.gumbel = gumbel
        self.pooling = pooling
        self.head = head

        # Matrizes de peso para cada Head
        self.W_u = nn.Parameter(torch.randn(head, dim, dim))
        self.W_i = nn.Parameter(torch.randn(head, dim, dim))
        self.M = nn.Parameter(torch.randn(head, dim, dim))
        self.reset_para()

    def reset_para(self):
        nn.init.xavier_uniform_(self.M)
        nn.init.xavier_uniform_(self.W_u)
        nn.init.xavier_uniform_(self.W_i)

    def forward(self, u_fea, i_fea):
        # u_fea: [B, H, L, D]
        u = torch.matmul(u_fea, self.W_u)
        i = torch.matmul(i_fea, self.W_i)

        # Matriz de afinidade: [B, H, L, L]
        S = u.matmul(self.M).matmul(i.transpose(-1, -2))

        if self.pooling == 'max':
            u_score, i_score = S.max(-1)[0], S.max(-2)[0]
        else:
            u_score, i_score = S.mean(-1), S.mean(-2)

        if self.gumbel:
            p_u = F.gumbel_softmax(u_score, hard=True, dim=-1)
            p_i = F.gumbel_softmax(i_score, hard=True, dim=-1)
        else:
            p_u = F.softmax(u_score, dim=-1)
            p_i = F.softmax(i_score, dim=-1)

        return p_u.unsqueeze(-1), p_i.unsqueeze(-1)