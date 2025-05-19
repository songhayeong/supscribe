import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionClassifier(nn.Module):
    def __init__(self, cat_dims, embed_dim=32, num_heads=4, num_layers=2, num_numeric=0):
        super().__init__()
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in cat_dims
        ])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_scores = []  # for visualization hook
        self._register_hooks()

        total_cat_dim = len(cat_dims) * embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_cat_dim + num_numeric, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _register_hooks(self):
        def hook(module, input, output):
            if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'attn_output_weights'):
                self.attn_scores.append(module.self_attn.attn_output_weights.detach().cpu())
        for layer in self.transformer.layers:
            layer.register_forward_hook(hook)

    def forward(self, x_cat, x_num=None):
        self.attn_scores = []
        x_cat_embed = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeds)], dim=1)
        x_cat_transformed = self.transformer(x_cat_embed)  # [B, T, D]
        x_cat_flat = x_cat_transformed.flatten(start_dim=1)  # [B, T*D]

        if x_num is not None:
            x = torch.cat([x_cat_flat, x_num], dim=1)
        else:
            x = x_cat_flat

        logits = self.classifier(x)
        return {
            "logits": logits,
            "attn_scores": self.attn_scores
        }