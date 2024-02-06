from __future__ import annotations

import pandas as pd

from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


def dataset_stats(data):
    return data.mean(), data.std(), data.min(), data.max()


def str_contains(df, col, pattern):
    mask = df[col].str.contains(pattern, na=False)
    return df[mask]


def check_for_missing_dates(df):
    date_range = pd.date_range(
        start=df.date.min(), end=df.date.max(), freq='D')
    missing_dates = date_range.difference(df.date)
    return missing_dates


def pressure_to_kPa(df1, df2):
    df1[['avg_sea_level_pres_hpa']] = df1[['avg_sea_level_pres_hpa']] / 10
    df2[['pressure']] = df2[['pressure']] / 1000
    return df1, df2


def n_nans(df):

    for col in df.columns:
        df_nans = df[col].isna().sum()
        print(f'{col}: {df_nans}, {round(((df_nans/len(df))*100), 3)}')
