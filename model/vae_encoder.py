# model/vae_encoder.py

import torch
import torch.nn as nn


class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),  # Dropout or BN 고려
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = self.mlp(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar