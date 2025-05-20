# data/dataset.py

import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from data.preprocessing import preprocess_ivf_dataframe


class IVFStrategyDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df = preprocess_ivf_dataframe(df)
        self.df = df # 시술 유형 등 메타 정보 접근용 저장

        label_col = '임신 성공 여부'
        if label_col not in df.columns:
            raise ValueError(f"'{label_col}' 컬럼이 데이터에 존재하지 않습니다.")

        # 컬럼 분리
        self.cat_cols = [col for col in df.select_dtypes(include='object').columns if col != label_col]
        self.num_cols = df.select_dtypes(include='number').drop(label_col, errors='ignore').columns.tolist()

        # 인코더 적용
        self.enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.scaler = StandardScaler()

        self.cat_encoded = self.enc.fit_transform(df[self.cat_cols]).astype('int64')
        self.num_scaled = self.scaler.fit_transform(df[self.num_cols]).astype('float32')
        self.y = df[label_col].values.astype('float32')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_cat = torch.tensor(self.cat_encoded[idx], dtype=torch.long)
        x_num = torch.tensor(self.num_scaled[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x_cat, x_num, y
