# data/missing_handler.py

import pandas as pd


def handle_missing_values(df: pd.DataFrame,
                          numeric_groups: dict,
                          categorical_columns: list = [],
                          nominal_columns: list = []) -> pd.DataFrame:
    """
    결측치 처리 함수:
    - 수치형 변수 그룹: 중앙값으로 채움
    - 범주형 변수: 최빈값
    - 명목형 변수: "NaN" 문자열로 채움
    """
    for group_name, columns in numeric_groups.items():
        for col in columns:
            if col in df.columns:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)

    for col in categorical_columns:
        if col in df.columns:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value).astype(str)

    for col in nominal_columns:
        if col in df.columns:
            df[col] = df[col].fillna("NaN").astype(str)

    return df
