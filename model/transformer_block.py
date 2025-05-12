# model/transformer_block.py

import torch
import torch.nn as nn


class TabTransformerBlock(nn.Module):
    def __init__(self, dim_embed, num_heads, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embed,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_cat_embed):
        return self.transformer(x_cat_embed)    # shape : (B, num_cat, dim_embed)
