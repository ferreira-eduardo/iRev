import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mask_to_key_padding(mask: torch.Tensor | None) -> torch.Tensor | None:
    """
    TF code passes enc/dec padding masks. In TF Transformers this is commonly shaped:
      - (B, 1, 1, S) with 1 for PAD positions
      - or (B, S) already
    PyTorch MultiheadAttention expects key_padding_mask: (B, S) with True for PAD.
    """
    if mask is None:
        return None

    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)

    # Convert to boolean "is_pad"
    if mask.dtype != torch.bool:
        mask = mask > 0

    if mask.dim() == 4:
        # (B,1,1,S) -> (B,S)
        mask = mask[:, 0, 0, :]
    elif mask.dim() == 3:
        # (B,1,S) -> (B,S)
        mask = mask[:, 0, :]
    elif mask.dim() == 2:
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    return mask  # (B,S) bool


class Interacter(nn.Module):
    """
    Minimal Transformer-style block that works for both:
      - self-attention: q=k=v
      - cross-attention: q != k=v
    Returns (out, attn_weights) where attn_weights has shape (B, H, Tq, Tk).
    """

    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers

        self.attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model, eps=1e-6) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model, eps=1e-6) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dff),
                nn.ReLU(),
                nn.Linear(dff, d_model),
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        q: torch.Tensor,  # (B, Tq, D)
        k: torch.Tensor,  # (B, Tk, D)
        v: torch.Tensor,  # (B, Tk, D)
        padding_mask: torch.Tensor | None = None,  # TF-style mask ok
    ):
        key_padding_mask = _mask_to_key_padding(padding_mask)  # (B, Tk) True=PAD

        last_attn = None
        x = q
        for l in range(self.num_layers):
            attn_out, attn_w = self.attn[l](
                x, k, v,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,  # keep per-head weights
            )
            x = self.norm1[l](x + self.dropout(attn_out))
            ffn_out = self.ffn[l](x)
            x = self.norm2[l](x + self.dropout(ffn_out))
            last_attn = attn_w  # (B, H, Tq, Tk)

        return x, last_attn


class KA(nn.Module):
    """
    PyTorch port of TF KA model in KANN.py.
    Notes:
      - Loads memmap embeddings ONCE in __init__ (TF code loads every forward; that’s very slow).
      - prediction_u_layer / prediction_i_layer use LazyLinear so they infer seq_len at first forward,
        mirroring TF Dense that auto-builds on first call.
    """

    def __init__(
        self,
        user_num: int,
        item_num: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        rate: float = 0.0,
        data_dir: str = "../data/imdb",
        re_file: str = "entity_emb50_concat_mean.bin",
        ori_file: str = "entity_oriemb50.bin",
        freeze_entity_emb: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.attn_dim = 4 * d_model

        # --- biases ---
        self.user_bias = nn.Embedding(user_num + 2, 1)
        self.item_bias = nn.Embedding(item_num + 2, 1)
        nn.init.constant_(self.user_bias.weight, 0.1)
        nn.init.constant_(self.item_bias.weight, 0.1)
        self.global_bias = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # --- load entity embeddings from memmap (same files as TF) ---
        re_path = os.path.join(data_dir, re_file)
        ori_path = os.path.join(data_dir, ori_file)

        # re_vec50: (N, 3*d_model)
        re_vec = np.memmap(re_path, dtype="float32", mode="r")
        col_re = d_model * 3
        n_re = int(re_vec.shape[0] / col_re)
        re_vec = np.resize(re_vec, (n_re, col_re))
        re_weight = torch.from_numpy(np.array(re_vec, copy=True))  # copy out of memmap

        # ori_e50: (N, d_model)
        ori_vec = np.memmap(ori_path, dtype="float32", mode="r")
        col_ori = d_model
        n_ori = int(ori_vec.shape[0] / col_ori)
        ori_vec = np.resize(ori_vec, (n_ori, col_ori))
        ori_weight = torch.from_numpy(np.array(ori_vec, copy=True))

        self.re_emb = nn.Embedding.from_pretrained(re_weight, freeze=freeze_entity_emb)
        self.ori_emb = nn.Embedding.from_pretrained(ori_weight, freeze=freeze_entity_emb)

        # --- norms/dropout ---
        self.layernorm_conq = nn.LayerNorm(self.attn_dim, eps=1e-6)
        self.layernorm_conk = nn.LayerNorm(self.attn_dim, eps=1e-6)
        self.dropout = nn.Dropout(rate)

        self.attention_u = Interacter(num_layers, self.attn_dim, num_heads, dff, rate)
        self.attention_i = Interacter(num_layers, self.attn_dim, num_heads, dff, rate)
        self.interacter_u = Interacter(num_layers, self.attn_dim, num_heads, dff, rate)
        self.interacter_i = Interacter(num_layers, self.attn_dim, num_heads, dff, rate)

        self.layernorm_con_oriu = nn.LayerNorm(5 * d_model, eps=1e-6)
        self.layernorm_con_orii = nn.LayerNorm(5 * d_model, eps=1e-6)

        # After interaction: (B, 5*d) -> (B, d)
        self.after_inter_layer = nn.Linear(5 * d_model, d_model)

        # These act on last dim after transpose: (B, 5*d, S) -> (B, 5*d, 1)
        self.prediction_u_layer = nn.LazyLinear(1)
        self.prediction_i_layer = nn.LazyLinear(1)

        self.prediction_layer = nn.Linear(d_model, 1)

    def forward(
        self,
        u_batch: torch.Tensor,          # (B, Su) long
        i_batch: torch.Tensor,          # (B, Si) long
        uid: torch.Tensor,              # (B,) long
        iid: torch.Tensor,              # (B,) long
        enc_padding_mask=None,
        dec_padding_mask=None,
    ):
        # --- embeddings (TF: embedding_lookup) ---
        q_context = self.re_emb(u_batch)  # (B, Su, 3d)
        k_context = self.re_emb(i_batch)  # (B, Si, 3d)
        q_ori = self.ori_emb(u_batch)     # (B, Su, d)
        k_ori = self.ori_emb(i_batch)     # (B, Si, d)

        # concat => 4d, layernorm, scale, dropout
        ureview_emb = self.layernorm_conq(torch.cat([q_context, q_ori], dim=2))  # (B, Su, 4d)
        ireview_emb = self.layernorm_conk(torch.cat([k_context, k_ori], dim=2))  # (B, Si, 4d)

        scale = math.sqrt(self.attn_dim)
        ureview_emb = self.dropout(ureview_emb * scale)
        ireview_emb = self.dropout(ireview_emb * scale)

        attention_weights = {}

        # self-attn
        u_out, attn_u = self.attention_u(ureview_emb, ureview_emb, ureview_emb, enc_padding_mask)
        i_out, attn_i = self.attention_i(ireview_emb, ireview_emb, ireview_emb, dec_padding_mask)

        # cross-attn
        inter_out_u, attn_ui = self.interacter_u(u_out, i_out, i_out, dec_padding_mask)
        inter_out_i, attn_iu = self.interacter_i(i_out, u_out, u_out, enc_padding_mask)

        attention_weights["attn_u"] = attn_u
        attention_weights["attn_i"] = attn_i
        attention_weights["attn_ui"] = attn_ui
        attention_weights["attn_iu"] = attn_iu

        # concat original features again => 5d
        inter_out_u = self.layernorm_con_oriu(torch.cat([inter_out_u, q_ori], dim=2))  # (B, Su, 5d)
        inter_out_i = self.layernorm_con_orii(torch.cat([inter_out_i, k_ori], dim=2))  # (B, Si, 5d)

        # transpose to apply Dense over sequence length (same as TF)
        inter_out_u = inter_out_u.transpose(1, 2)  # (B, 5d, Su)
        inter_out_i = inter_out_i.transpose(1, 2)  # (B, 5d, Si)

        out_u = F.relu(self.prediction_u_layer(inter_out_u)).squeeze(-1)  # (B, 5d)
        out_i = F.relu(self.prediction_i_layer(inter_out_i)).squeeze(-1)  # (B, 5d)

        interaction = out_u * out_i  # (B, 5d)
        interaction = F.relu(self.after_inter_layer(interaction))  # (B, d_model)

        scores = F.relu(self.prediction_layer(interaction))  # (B, 1)

        user_b = self.user_bias(uid)  # (B, 1)
        item_b = self.item_bias(iid)  # (B, 1)

        scores = scores + user_b + item_b + self.global_bias
        return scores, attention_weights
