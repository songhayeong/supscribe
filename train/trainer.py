# train/trainer.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def loss_function(x_hat, x, y_hat, y, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(x_hat, x, reduction='mean')
    cls_loss = F.binary_cross_entropy(y_hat, y.unsqueeze(1))
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, cls_loss, kl_div, recon_loss + cls_loss + beta * kl_div


def train_epoch(model, dataloader, optimizer, device='cpu'):
    model.train()
    total_loss, total_recon, total_cls, total_kl = 0, 0, 0, 0

    for x_cat, x_num, y in tqdm(dataloader, desc="Training", leave=False):
        x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)

        out = model(x_cat, x_num)
        x_target = out['x_target']  # <-- forward에서 반환된 타겟 사용

        recon, cls, kl, loss = loss_function(
            out['x_hat'], x_target, out['y_hat'], y, out['mu'], out['logvar']
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_cls += cls.item()
        total_kl += kl.item()

    return {
        'loss': total_loss / len(dataloader),
        'recon': total_recon / len(dataloader),
        'cls': total_cls / len(dataloader),
        'kl': total_kl / len(dataloader)
    }
