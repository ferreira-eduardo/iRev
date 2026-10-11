# InteracterLayer.py (PyTorch version)
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_key_padding_mask(mask):
    """
    TF-style padding mask often comes as:
      - (B, 1, 1, S) with 1/True meaning "mask out"
      - (B, S) with 1/True meaning "mask out"
    PyTorch MultiheadAttention expects key_padding_mask: (B, S) bool where True means "ignore".
    """
    if mask is None:
        return None
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)

    if mask.dtype != torch.bool:
        mask = mask > 0

    if mask.dim() == 4:      # (B,1,1,S)
        mask = mask[:, 0, 0, :]
    elif mask.dim() == 3:    # (B,1,S)
        mask = mask[:, 0, :]
    elif mask.dim() == 2:    # (B,S)
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    return mask  # (B,S) bool


class PointWiseFFN(nn.Module):
    """Equivalent to TF point_wise_feed_forward_network(d_model, dff)."""
    def __init__(self, d_model: int, dff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dff)
        self.fc2 = nn.Linear(dff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class InteracterLayer(nn.Module):
    """
    PyTorch port of the provided TF InteracterLayer.

    TF behavior:
      attn_output, attn = MHA(q,k,v,mask)
      out1 = LN(q + Dropout(attn_output))
      out2 = LN(out1 + Dropout(FFN(out1)))
      return out2, attn
    """
    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float):
        super().__init__()

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=rate,
            batch_first=True,
        )
        self.ffn = PointWiseFFN(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, q, k, v, training=None, mask=None):
        # Match TF signature: training can be passed; otherwise use module state.
        if training is None:
            training = self.training

        key_padding_mask = _to_key_padding_mask(mask)  # (B, Tk) True=ignore

        attn_output, attn = self.mha(
            q, k, v,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # keep per-head weights (B,H,Tq,Tk)
        )

        if training:
            attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(q + attn_output)

        ffn_output = self.ffn(out1)
        if training:
            ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2, attn
