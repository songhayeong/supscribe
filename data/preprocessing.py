# data/preprocessing.py

import numpy as np
import pandas as pd


def preprocess_ivf_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns='ID')

    """
    순수 전처리: 결측치, 형식 통일 등만 처리하고 인코딩은 하지 않음
    """
    # 타겟 이진화
    if '임신 성공 여부' in df.columns:
        df['임신 성공 여부'] = df['임신 성공 여부'].apply(lambda x: 1 if x > 0 else 0)

    # 횟수형 문자열 → 정수
    count_columns = [
        "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수",
        "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
        "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수"
    ]

    def convert_count(value):
        if isinstance(value, str) and "회 이상" in value:
            return 6
        try:
            return int(str(value).replace("회", ""))
        except:
            return np.nan

    for col in count_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(convert_count)

    # 시술 당시 나이
    age_mapping = {
        "만18-34세": 25, "만35-37세": 36, "만38-39세": 38, "만40-42세": 41,
        "만43-44세": 43, "만45-50세": 44, "알 수 없음": np.nan
    }
    if "시술 당시 나이" in df.columns:
        df["시술 당시 나이 (변환)"] = df["시술 당시 나이"].map(age_mapping)

    # 이진 수치형 → object 변환 (0/1 범주 처리)
    for col in df.select_dtypes(include='number').columns:
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1}):
            df[col] = df[col].astype('object')

    # 문자열 결측치 채움
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).fillna("missing")

    # 수치형 결측치 중앙값 대체
    for col in df.select_dtypes(include='number').columns.difference(['임신 성공 여부']):
        df[col] = df[col].fillna(df[col].median())

    # 나이 보간
    if "시술 당시 나이 (변환)" in df.columns:
        df["시술 당시 나이 (변환)"] = df["시술 당시 나이 (변환)"].fillna(df["시술 당시 나이 (변환)"].mean())

    return df
