import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_analysis_model.model.custom_transformer_encoder import CustomTransformerEncoderLayer


class AttentionClassifier(nn.Module):
    def __init__(self, cat_dims, embed_dim=32, num_heads=4, num_layers=2, num_numeric=0):
        super().__init__()
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in cat_dims
        ])

        # ← Custom 레이어로 변경
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_scores = []
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
            if hasattr(module, 'attn_output_weights') and module.attn_output_weights is not None:
                self.attn_scores.append(module.attn_output_weights.detach().cpu())

        for layer in self.transformer.layers:
            layer.register_forward_hook(hook)

    def forward(self, x_cat, x_num=None):
        self.attn_scores = []
        x_cat_embed = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeds)], dim=1)
        x_cat_transformed = self.transformer(x_cat_embed)
        x_cat_flat = x_cat_transformed.flatten(start_dim=1)

        if x_num is not None:
            x = torch.cat([x_cat_flat, x_num], dim=1)
        else:
            x = x_cat_flat

        logits = self.classifier(x)
        return {
            "logits": logits,
            "attn_scores": self.attn_scores
        }