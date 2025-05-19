import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# ---------------------------------------------
# 1. Load attention pair CSVs
csv_dir = "../data/attn_csv"  # ← 경로 맞게 수정
attention_vectors = []
procedure_labels = []
feature_names = None

for fname in os.listdir(csv_dir):
    if fname.endswith(".csv"):
        path = os.path.join(csv_dir, fname)
        df = pd.read_csv(path)
        sorted_df = df.sort_values(["query", "key"])
        vector = sorted_df["attention_score"].values.tolist()
        attention_vectors.append(vector)

        label = fname.replace("attention_pairs_", "").replace(".csv", "")
        procedure_labels.append(label)

        if feature_names is None:
            feature_names = [f"{q}→{k}" for q, k in zip(sorted_df["query"], sorted_df["key"])]

attn_df = pd.DataFrame(attention_vectors, index=procedure_labels, columns=feature_names)
attn_scaled = StandardScaler().fit_transform(attn_df)

# ---------------------------------------------
# 2. KMeans Clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(attn_scaled)

# ---------------------------------------------
# 3. PCA for 3D Visualization
pca = PCA(n_components=3)
attn_pca = pca.fit_transform(attn_scaled)

# ---------------------------------------------
# 4. Visualization (3D scatter)
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
palette = sns.color_palette("tab10", n_clusters)
colors = [palette[label] for label in labels]

scatter = ax.scatter(
    attn_pca[:, 0], attn_pca[:, 1], attn_pca[:, 2],
    c=colors, s=100, alpha=0.9
)

# 시술 이름 + 클러스터 번호 표시
for i, label in enumerate(procedure_labels):
    ax.text(
        attn_pca[i, 0], attn_pca[i, 1], attn_pca[i, 2],
        f"{label}\n(C{labels[i]})", fontsize=8
    )

# 범례 추가
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=f"Cluster {i}",
           markerfacecolor=palette[i], markersize=10)
    for i in range(n_clusters)
]
ax.legend(handles=legend_elements, loc='upper right')

ax.set_title("3D PCA + KMeans Clustering of Treatment Attention Vectors", fontsize=14)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.tight_layout()
plt.show()
# ---------------------------------------------
# 5. Print cluster-wise treatment groups
print("📌 클러스터별 시술 유형 리스트:")
for cluster_id in range(n_clusters):
    print(f"\n[Cluster {cluster_id}]")
    for i, label in enumerate(procedure_labels):
        if labels[i] == cluster_id:
            print(" -", label)

# ---------------------------------------------
# 6. Top 10 contributing attention pairs per cluster
print("\n📌 클러스터별 중심 attention vector의 상위 10개 feature:")
for cluster_id in range(n_clusters):
    indices = [i for i, l in enumerate(labels) if l == cluster_id]
    center = attn_df.iloc[indices].mean().values
    topk_idx = np.argsort(center)[::-1][:10]
    print(f"\n[Cluster {cluster_id}]")
    for j in topk_idx:
        print(f" - {attn_df.columns[j]}: {center[j]:.4f}")