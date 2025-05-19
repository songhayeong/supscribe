import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

mpl.rcParams['font.family'] = 'AppleGothic'  # 또는 'NanumGothic'
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 깨짐 방지

# CSV 파일이 들어 있는 디렉토리 경로
csv_dir = "../data/attn_csv"  # 필요 시 "data/attn_csv" 또는 "../data/attn_csv" 등으로 수정

# 시술 유형별 attention 벡터 수집
attention_vectors = []
procedure_labels = []

for fname in os.listdir(csv_dir):
    if fname.endswith(".csv"):
        path = os.path.join(csv_dir, fname)
        df = pd.read_csv(path)
        sorted_df = df.sort_values(["query", "key"])
        vector = sorted_df["attention_score"].values.tolist()
        attention_vectors.append(vector)
        label = fname.replace("attention_pairs_", "").replace(".csv", "")
        procedure_labels.append(label)

# DataFrame 생성 및 정규화
attn_df = pd.DataFrame(attention_vectors, index=procedure_labels)
attn_scaled = StandardScaler().fit_transform(attn_df)

# PCA 분석
pca = PCA(n_components=5)
pca.fit(attn_scaled)

# feature 이름 (query→key 조합)
example_df = pd.read_csv(os.path.join(csv_dir, os.listdir(csv_dir)[0]))
sorted_example = example_df.sort_values(["query", "key"])
feature_names = [f"{q}→{k}" for q, k in zip(sorted_example["query"], sorted_example["key"])]

# PC별 feature 기여도 (loadings)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=feature_names
)

# 시각화: PC1, PC2의 주요 기여 feature 15개씩
for i in range(2):  # PC1, PC2
    pc_col = f"PC{i+1}"
    top_features = loadings[pc_col].abs().sort_values(ascending=False).head(15).index
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=loadings.loc[top_features, pc_col],
        y=top_features,
        palette="viridis"
    )
    plt.title(f"Top 15 Feature Contributions to {pc_col}")
    plt.xlabel("Loading Value")
    plt.ylabel("Feature (query → key)")
    plt.tight_layout()
    plt.show()