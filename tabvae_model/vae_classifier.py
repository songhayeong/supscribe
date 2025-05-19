# tabvae_model/vae_classifier.py

import torch
import torch.nn as nn


class VAEClassifier(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)
