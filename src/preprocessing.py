import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Optional, Union
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def load_raw_data(
    train_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
):
    """Загрузка сырых данных"""
    if train_path is None:
        train_path = BASE_DIR / "data" / "hackathon_income_train.csv"
    else:
        train_path = Path(train_path)

    if test_path is None:
        test_path = BASE_DIR / "data" / "hackathon_income_test.csv"
    else:
        test_path = Path(test_path)

    train_df = pd.read_csv(train_path, sep=';')
    test_df = pd.read_csv(test_path, sep=';')
    return train_df, test_df


def load_test_df(test_path: Optional[Union[str, Path]] = None):
    """Загрузка только test_df"""
    if test_path is None:
        test_path = BASE_DIR / "data" / "hackathon_income_test.csv"
    else:
        test_path = Path(test_path)

    test_df = pd.read_csv(test_path, sep=';', decimal=',')
    return test_df

def preprocess_full_pipeline(df_train, df_test):
    """Полная предобработка"""

    exclude_cols = ["id", "target", "w", "dt"]
    num_cols = df_train.select_dtypes(include=["float64", "int64"]).columns.difference(exclude_cols)
    cat_cols = df_train.select_dtypes(include=["object"]).columns.tolist()

    full_df = pd.concat([df_train.drop(columns=["target", "w"]), df_test], axis=0, ignore_index=True)
    train_size = df_train.shape[0]

    full_proc = full_df.copy()

    for col in num_cols:
        median_val = full_proc[col].median()
        full_proc[col] = full_proc[col].fillna(median_val)

    for col in cat_cols:
        full_proc[col] = full_proc[col].astype(str).fillna("missing")

    na_frac = full_df.isna().mean()
    drop_cols = na_frac[na_frac > 0.9].index.tolist()
    drop_cols = [col for col in drop_cols if col not in ["id", "target", "w"]]

    full_proc = full_proc.drop(columns=drop_cols)
    num_cols = num_cols.difference(drop_cols)
    cat_cols = [col for col in cat_cols if col not in drop_cols]

    X_train_final = full_proc.iloc[:train_size].copy()
    X_test_final = full_proc.iloc[train_size:].copy()


    return X_train_final, X_test_final, num_cols.tolist(), cat_cols


def preprocess_single_client(raw_client_df, num_cols, cat_cols):
    """Предобработка одного клиента из test_df для модели"""
    client_proc = raw_client_df.copy()

    # Предобработка числовых колонок
    for col in num_cols:
        if col in client_proc.columns:

            if 'dt' in col or 'date' in col:
                client_proc[col] = pd.to_datetime(client_proc[col], errors='coerce').astype('int64') // 10**9
            else:
                median_val = client_proc[col].median() if client_proc[col].notna().sum() > 0 else 0
                client_proc[col] = client_proc[col].fillna(median_val)

    # Категориальные колонки
    for col in cat_cols:
        if col in client_proc.columns:
            client_proc[col] = client_proc[col].astype(str).fillna("missing")

    # Убираем колонки с >90% NA
    na_frac = client_proc.isna().mean()
    drop_cols = na_frac[na_frac > 0.9].index.tolist()
    client_proc = client_proc.drop(columns=drop_cols)


    for col in client_proc.select_dtypes('number').columns:
        client_proc[col] = pd.to_numeric(client_proc[col], errors='coerce').fillna(0)

    return client_proc


def add_engineered_features(client_proc):
    """Добавляем new_features"""
    new_features = {}

    key_features = ['hdb_outstand_sum', 'turn_cur_cr_sum_v2', 'turn_cur_db_sum_v2',
                    'curr_rur_amt_3m_avg', 'salary_6to12m_avg', 'dp_ils_avg_salary_1y',
                    'first_salary_income', 'turn_cur_cr_7avg_avg_v2', 'turn_cur_cr_avg_v2']

    for feat_name, feat_expr in {
        'debt_to_turnover': lambda df: df.get('hdb_outstand_sum', 0) / (df.get('turn_cur_cr_sum_v2', 1) + 1),
        'debit_cr_ratio': lambda df: df.get('turn_cur_db_sum_v2', 0) / (df.get('turn_cur_cr_sum_v2', 1) + 1),
        'salary_stability': lambda df: df.get('salary_6to12m_avg', 0) / (df.get('dp_ils_avg_salary_1y', 1) + 1),
        'has_salary_flag': lambda df: (df.get('first_salary_income', 0) > 0).astype(int)
    }.items():
        new_features[feat_name] = feat_expr(client_proc)

    # Добавляем к клиенту
    for name, value in new_features.items():
        client_proc[name] = value

    return client_proc

def align_client_features(client_proc: pd.DataFrame, model_features: list[str]):
    """
    Добавляем отсутствующие колонки с нулями и оставляем только нужные для модели.
    """
    for col in model_features:
        if col not in client_proc.columns:
            client_proc[col] = 0
    client_proc = client_proc[model_features]
    return client_proc

def preprocess_client_for_model(client_id, test_df, num_cols, cat_cols):
    """client_id - готовые данные для модели"""
    raw_client = test_df[test_df['id'] == client_id].copy()
    if raw_client.empty:
        raise ValueError(f"Клиент {client_id} не найден!")

    client_proc = preprocess_single_client(raw_client, num_cols, cat_cols)
    client_final = add_engineered_features(client_proc)

    return client_final.iloc[0:1]

def save_preprocessing_info(num_cols, cat_cols, filepath="preprocessing_info.pkl"):
    joblib.dump({
        'num_features': num_cols,
        'cat_features': cat_cols
    }, filepath)
