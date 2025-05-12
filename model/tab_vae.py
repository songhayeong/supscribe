import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer_block import TabTransformerBlock
from model.vae_encoder import VAEEncoder
from model.vae_decoder import VAEDecoder
from model.vae_classifier import VAEClassifier


class TabVAEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        cat_dims = cfg['model']['categorical_dims']
        self.num_cat = len(cat_dims)
        self.dim_embed = cfg['model']['embed_dim']
        self.latent_dim = cfg['model']['latent_dim']
        self.input_dim_num = cfg['model']['num_numeric']

        self.cat_embeds = nn.ModuleList([
            nn.Embedding(num_cat, self.dim_embed) for num_cat in cat_dims
        ])

        self.transformer = TabTransformerBlock(
            dim_embed=self.dim_embed,
            num_heads=cfg['model']['num_heads'],
            num_layers=cfg['model']['num_layers']
        )

        total_embed_dim = self.num_cat * self.dim_embed + self.input_dim_num

        self.encoder = VAEEncoder(total_embed_dim, self.latent_dim)
        self.decoder = VAEDecoder(self.latent_dim, total_embed_dim)
        self.classifier = VAEClassifier(self.latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_cat, x_num):
        x_cat_embed = torch.stack([embed(x_cat[:, i]) for i, embed in enumerate(self.cat_embeds)], dim=1)
        x_cat_encoded = self.transformer(x_cat_embed)
        x_cat_flat = x_cat_encoded.flatten(start_dim=1)

        x_full = torch.cat([x_cat_flat, x_num], dim=1)
        mu, logvar = self.encoder(x_full)
        z = self.reparameterize(mu, logvar)

        x_hat = self.decoder(z)     # 얘는 해석가능성 확보 및 latent에서 나온거를 복원해서 reconstruction error check!
        y_hat = self.classifier(z)  # 여기서 classifier는 decoder에서 나온것으로 할 것인지 ? 는 생각해봐야한다.

        return {
            'x_hat': x_hat,
            'y_hat': y_hat,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'x_target': x_full
        }