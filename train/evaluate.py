# train/evaluate.py

import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from train.trainer import loss_function
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns




def evaluate(model, dataloader, device='cpu'):
    model.eval()
    y_true, y_prob, y_pred = [], [], []
    total_loss, total_recon, total_cls, total_kl = 0, 0, 0, 0

    first_batch_done = False

    with torch.no_grad():
        for x_cat, x_num, y in dataloader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)

            out = model(x_cat, x_num)
            x_target = out['x_target']  # <-- forward에서 반환된 타겟 사용

            recon, cls, kl, loss = loss_function(
                out['x_hat'], x_target, out['y_hat'], y, out['mu'], out['logvar']
            )

            total_loss += loss.item()
            total_recon += recon.item()
            total_cls += cls.item()
            total_kl += kl.item()

            y_true.extend(y.cpu().numpy())
            y_prob.extend(out['y_hat'].cpu().numpy())
            y_pred.extend((out['y_hat'] > 0.5).int().cpu().numpy())

            feature_names = getattr(dataloader.dataset, 'cat_cols', None)

            # 첫 배치에 대한 시각화
            if not first_batch_done:
                mpl.rcParams['font.family'] = 'AppleGothic'
                mpl.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

                first_batch_done = True
                attn_scores = model.transformer.attn_scores
                print(f"[DEBUG] Number of layers with attn: {len(attn_scores)}")

                if attn_scores:
                    attn = attn_scores[0]  # Layer 1
                    print(f"[DEBUG] attn shape: {attn.shape}")  # Expect [B, H, T, T]
                    head = attn[0, 0]  # batch 0, head 0

                    plt.figure(figsize=(8, 6))
                    sns.heatmap(head, cmap="viridis",
                                xticklabels=feature_names if feature_names else None,
                                yticklabels=feature_names if feature_names else None)
                    plt.title("Attention Map: Layer 1, Head 1 (First Batch)")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()

    return {
        'loss': total_loss / len(dataloader),
        'recon': total_recon / len(dataloader),
        'cls': total_cls / len(dataloader),
        'kl': total_kl / len(dataloader),
        'auc': roc_auc_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred),
        'acc': accuracy_score(y_true, y_pred)
    }
