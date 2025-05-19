import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from train.trainer import loss_function
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from collections import defaultdict


def evaluate(model, dataloader, device='cpu'):
    model.eval()
    y_true, y_prob, y_pred = [], [], []
    total_loss, total_recon, total_cls, total_kl = 0, 0, 0, 0

    first_batch_done = False
    attn_by_procedure = defaultdict(list)
    procedure_types = dataloader.dataset.df["특정 시술 유형"].values

    with torch.no_grad():
        for idx, (x_cat, x_num, y) in enumerate(dataloader):
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

                    plt.figure(figsize=(20, 18))
                    sns.heatmap(head, cmap="viridis",
                                xticklabels=feature_names if feature_names else None,
                                yticklabels=feature_names if feature_names else None,
                                cbar_kws={'shrink': 0.8})
                    plt.title("Attention Map: Layer 1, Head 1 (First Batch)", fontsize=24)
                    plt.xticks(rotation=90, fontsize=11)
                    plt.yticks(fontsize=11)
                    plt.tight_layout()
                    plt.show()

            # Collect attention scores by procedure type for all batches
            if model.transformer.attn_scores:
                attn = model.transformer.attn_scores[0]  # Layer 1
                # attn shape: [B, H, T, T]
                # We'll average over heads for each sample
                attn_avg_heads = attn[:, :, :, :].mean(dim=1)  # [B, T, T]

                # For each sample in batch
                batch_start_idx = idx * dataloader.batch_size
                for i in range(attn_avg_heads.size(0)):
                    proc_idx = batch_start_idx + i
                    if proc_idx >= len(procedure_types):
                        continue
                    proc_type = procedure_types[proc_idx]
                    attn_by_procedure[proc_type].append(attn_avg_heads[i].cpu())

    # After loop, plot average attention maps per procedure type
    if attn_by_procedure:
        mpl.rcParams['font.family'] = 'AppleGothic'
        mpl.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

        for proc_type, attn_list in attn_by_procedure.items():
            attn_tensor = torch.stack(attn_list, dim=0)  # [N, T, T]
            avg_attn = attn_tensor.mean(dim=0)

            plt.figure(figsize=(20, 18))
            sns.heatmap(avg_attn, cmap="viridis",
                        xticklabels=feature_names if feature_names else None,
                        yticklabels=feature_names if feature_names else None,
                        cbar_kws={'shrink': 0.8})
            plt.title(f"Average Attention Map: Layer 1, Head Avg ({proc_type})", fontsize=24)
            plt.xticks(rotation=90, fontsize=11)
            plt.yticks(fontsize=11)
            plt.tight_layout()
            plt.show()

            flat_attn = avg_attn.flatten()
            topk = torch.topk(flat_attn, 15)
            T = avg_attn.shape[0]
            print("\n[Top 15 Attention Pairs (query → key)]:")
            for idx, value in zip(topk.indices.tolist(), topk.values.tolist()):
                row = idx // T
                col = idx % T
                q_feat = feature_names[row] if feature_names else f"query_{row}"
                k_feat = feature_names[col] if feature_names else f"key_{col}"
                print(f"{q_feat} → {k_feat}: {value:.4f}")

            # Save top 15 attention pairs as CSV
            import csv
            import os

            save_dir = os.path.join("data", "attn_csv")
            os.makedirs(save_dir, exist_ok=True)

            safe_proc = proc_type.replace("/", "_").replace(" ", "_").replace(":", "-")
            csv_path = os.path.join(save_dir, f"attention_pairs_{safe_proc}.csv")

            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["query", "key", "attention_score"])
                for idx, value in zip(topk.indices.tolist(), topk.values.tolist()):
                    row = idx // T
                    col = idx % T
                    q_feat = feature_names[row] if feature_names else f"query_{row}"
                    k_feat = feature_names[col] if feature_names else f"key_{col}"
                    writer.writerow([q_feat, k_feat, round(value, 4)])
            print(f"[Saved] attention_pairs_{proc_type}.csv")

    return {
        'loss': total_loss / len(dataloader),
        'recon': total_recon / len(dataloader),
        'cls': total_cls / len(dataloader),
        'kl': total_kl / len(dataloader),
        'auc': roc_auc_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred),
        'acc': accuracy_score(y_true, y_pred)
    }
