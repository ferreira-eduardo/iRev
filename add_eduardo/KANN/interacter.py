# Interacter.py (PyTorch version)
import torch
import torch.nn as nn

from InteracterLayer import InteracterLayer  # expects your PyTorch InteracterLayer


class Interacter(nn.Module):
    """
    PyTorch port of the provided TensorFlow Interacter layer.

    - num_layers: number of InteracterLayer blocks.
    - Returns:
        x: last layer output
        attention_weights: dict with keys attention1..attentionN
    """

    def __init__(self, num_layers, d_model, num_heads, dff, rate):
        super().__init__()
        self.d_model = d_model
        self.enc_layers = nn.ModuleList(
            [InteracterLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )

    def forward(self, q, k, v, training=None, mask=None):
        # Keep the same API as TF: training can be passed explicitly;
        # otherwise rely on self.training.
        if training is None:
            training = self.training

        attention_weights = {}
        x = None
        for i, enc_layer in enumerate(self.enc_layers):
            x, attn = enc_layer(q, k, v, training=training, mask=mask)
            attention_weights[f"attention{i + 1}"] = attn

        return x, attention_weights
