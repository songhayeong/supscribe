import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 저장된 attention pair CSV가 있는 경로
csv_dir = "../data/attn_csv"  # 예: data/attn_csv/attention_pairs_ICSI_AH.csv 등

attention_vectors = []
procedure_labels = []

# 2. 각 시술 유형별 attention vector 불러오기
for fname in os.listdir(csv_dir):
    if not fname.endswith(".csv"):
        continue
    path = os.path.join(csv_dir, fname)
    df = pd.read_csv(path)

    # query-key 쌍 기준 정렬하여 feature vector 생성
    sorted_df = df.sort_values(["query", "key"])
    vector = sorted_df["attention_score"].values.tolist()

    attention_vectors.append(vector)
    label = fname.replace("attention_pairs_", "").replace(".csv", "")
    procedure_labels.append(label)

# 3. DataFrame 생성 및 정규화
attn_df = pd.DataFrame(attention_vectors, index=procedure_labels)
attn_scaled = StandardScaler().fit_transform(attn_df)

# 4. PCA + t-SNE 적용
pca = PCA(n_components=min(10, attn_scaled.shape[1]))
attn_pca = pca.fit_transform(attn_scaled)

tsne = TSNE(n_components=2, random_state=42, perplexity=5)
tsne_proj = tsne.fit_transform(attn_pca)

# 5. 시각화
plt.figure(figsize=(10, 7))
sns.scatterplot(x=tsne_proj[:, 0], y=tsne_proj[:, 1],
                hue=procedure_labels, s=120, palette="tab10")

plt.title("Clustering of Attention Structures per Treatment Type (t-SNE)", fontsize=16)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()