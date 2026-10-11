# MultiHeadAttention.py (PyTorch version)
import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    PyTorch port of the provided TF MultiHeadAttention.

    Output:
      - output: (B, seq_len_q, d_model)
      - attention_weights: (B, num_heads, seq_len_q, seq_len_k)

    Mask behavior (matches TF):
      - mask is added to logits as (mask * -1e9)
      - so mask should be broadcastable to (B, num_heads, seq_len_q, seq_len_k)
      - mask values: 1 for positions to mask out, 0 otherwise (same as your TF create_padding_mask)
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, d_model)
        return: (B, H, S, depth)
        """
        b, s, _ = x.shape
        x = x.view(b, s, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        q: (B, H, Sq, D)
        k: (B, H, Sk, D)
        v: (B, H, Sk, D)
        mask: broadcastable to (B, H, Sq, Sk), with 1 meaning "mask out".
        """
        # (B, H, Sq, Sk)
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_logits = matmul_qk / math.sqrt(q.size(-1))

        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask, device=scaled_logits.device)
            mask = mask.to(dtype=scaled_logits.dtype, device=scaled_logits.device)
            scaled_logits = scaled_logits + (mask * -1e9)

        attn_weights = torch.softmax(scaled_logits, dim=-1)          # (B,H,Sq,Sk)
        output = torch.matmul(attn_weights, v)                      # (B,H,Sq,D)
        return output, attn_weights

    def forward(self, q, k, v, mask=None):
        """
        q,k,v: (B, S, d_model)
        mask: broadcastable to (B, H, Sq, Sk)
        """
        # linear projections
        q = self.wq(q)  # (B, Sq, d_model)
        k = self.wk(k)  # (B, Sk, d_model)
        v = self.wv(v)  # (B, Sk, d_model)

        # split heads
        q = self.split_heads(q)  # (B, H, Sq, depth)
        k = self.split_heads(k)  # (B, H, Sk, depth)
        v = self.split_heads(v)  # (B, H, Sk, depth)

        # attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # concat heads: (B, Sq, H, depth) -> (B, Sq, d_model)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        b, sq, _, _ = scaled_attention.shape
        concat_attention = scaled_attention.view(b, sq, self.d_model)

        # final linear
        output = self.dense(concat_attention)  # (B, Sq, d_model)
        return output, attention_weights
