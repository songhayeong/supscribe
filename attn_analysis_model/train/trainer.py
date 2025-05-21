import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, device='cpu'):
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    y_true, y_pred, y_prob = [], [], []
    total_loss = 0

    loop = tqdm(dataloader, desc="Training", leave=False)  # tqdm 감싸기

    for x_cat, x_num, y in loop:
        x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)

        out = model(x_cat, x_num)
        logits = out["logits"].squeeze(1)

        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        y_true.extend(y.cpu().numpy())
        y_prob.extend(probs.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())

    return {
        "loss": total_loss / len(dataloader),
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob)
    }
