import torch
import torch.nn as nn
from tabvae_model.custom_transformer import CustomTransformerEncoderLayer


class TabTransformerBlock(nn.Module):
    def __init__(self, dim_embed, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=dim_embed,
                nhead=num_heads,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.attn_scores = []

    def forward(self, x):
        self.attn_scores.clear()
        for layer in self.layers:
            x = layer(x)
            self.attn_scores.append(layer.saved_attn_weights)  # head-wise attention
        return x
