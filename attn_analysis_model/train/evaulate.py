import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib as mpl

def evaluate(model, dataloader,
             device='cpu', feature_names=None):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    total_loss = 0

    criterion = torch.nn.BCEWithLogitsLoss()
    first_batch_done = False

    with torch.no_grad():
        for x_cat, x_num, y in dataloader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            out = model(x_cat, x_num)
            logits = out['logits'].squeeze(1)

            loss = criterion(logits, y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            y_true.extend(y.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # Attention 시각화 (첫 배치만)
            if not first_batch_done and out["attn_scores"]:
                first_batch_done = True
                attn_scores = out["attn_scores"][0]  # 첫 Layer
                head = attn_scores[0]  # 첫 sample, 첫 head

                mpl.rcParams['font.family'] = 'AppleGothic'
                mpl.rcParams['axes.unicode_minus'] = False
                plt.figure(figsize=(12, 10))
                sns.heatmap(head, cmap="viridis",
                            xticklabels=feature_names,
                            yticklabels=feature_names)
                plt.title("Attention Map (Layer 1, Head 1)")
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.show()

    return {
        'loss': total_loss / len(dataloader),
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob)
    }
