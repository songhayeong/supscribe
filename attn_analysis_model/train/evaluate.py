import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from collections import defaultdict
import csv
import os


def evaluate(model, dataloader, device='cpu'):
    model.eval()
    y_true, y_prob, y_pred = [], [], []
    first_batch_done = False
    attn_by_procedure = defaultdict(list)
    procedure_types = dataloader.dataset.df["특정 시술 유형"].values

    with torch.no_grad():
        for idx, (x_cat, x_num, y) in enumerate(dataloader):
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            out = model(x_cat, x_num)
            logits = out['logits'].squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()

            y_true.extend(y.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            feature_names = getattr(dataloader.dataset, 'cat_cols', None)

            if not first_batch_done:
                mpl.rcParams['font.family'] = 'AppleGothic'
                mpl.rcParams['axes.unicode_minus'] = False

                first_batch_done = True
                attn_scores = model.attn_scores
                print("[DEBUG] attn_scores length:", len(model.attn_scores))
                if model.attn_scores:
                    print("[DEBUG] attn_scores[0] shape:", model.attn_scores[0].shape)
                if attn_scores:
                    attn = attn_scores[0]                # [B, H, T, T]
                    head_avg = attn[0].mean(dim=0)       # [T, T] 평균

                    plt.figure(figsize=(20, 18))
                    sns.heatmap(head_avg, cmap="viridis",
                                xticklabels=feature_names if feature_names else None,
                                yticklabels=feature_names if feature_names else None,
                                cbar_kws={'shrink': 0.8})
                    plt.title("Attention Map: Layer 1, All Heads Avg (First Batch)", fontsize=24)
                    plt.xticks(rotation=90, fontsize=11)
                    plt.yticks(fontsize=11)
                    plt.tight_layout()
                    plt.show()

            if model.attn_scores:
                attn = model.attn_scores[0]              # [B, H, T, T]
                attn_avg_heads = attn.mean(dim=1)        # 평균 over heads → [B, T, T]

                batch_start_idx = idx * dataloader.batch_size
                for i in range(attn_avg_heads.size(0)):
                    proc_idx = batch_start_idx + i
                    if proc_idx >= len(procedure_types):
                        continue
                    proc_type = procedure_types[proc_idx]
                    attn_by_procedure[proc_type].append(attn_avg_heads[i].cpu())

    if attn_by_procedure:
        mpl.rcParams['font.family'] = 'AppleGothic'
        mpl.rcParams['axes.unicode_minus'] = False

        for proc_type, attn_list in attn_by_procedure.items():
            attn_tensor = torch.stack(attn_list, dim=0)
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
        'loss': 0.0,
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob)
    }
