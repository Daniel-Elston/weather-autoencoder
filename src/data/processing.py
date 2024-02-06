from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_data import RawDataLoader
from utils.my_utils import pressure_to_kPa
from utils.setup_env import setup_project_env
# from torch.utils.data import random_split

project_dir, config, setup_logs = setup_project_env()


class Processor(RawDataLoader):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def drop_cols(self, df):
        df = df.drop(['avg_wind_dir_deg', 'peak_wind_gust_kmh',
                     'sunshine_total_min'], axis=1)
        return df

    def fillna_from_df2(self, df1, df2, to_impute, impute_with):
        for df1_col, df2_col in zip(to_impute, impute_with):
            df1[df1_col] = df1[df1_col].fillna(df2[df2_col])
        return df1

    def process_dt_df1(self, df1):
        df1 = df1[df1['date'] >= '1973-01-01']

        df1['date_idx'] = pd.to_datetime(df1['date'], format='%Y%m%d')
        df1 = df1.set_index('date_idx')

        complete_index = pd.date_range(
            start=df1.index.min(), end=df1.index.max(), freq='D')
        df1 = df1.reindex(complete_index)

        df1 = df1.drop(columns=['date'])
        df1 = df1.reset_index(names='date')

        df1['date_idx'] = df1['date']
        df1 = df1.set_index('date_idx')
        return df1

    def process_dt_df2(self, df2):
        df2.date = pd.to_datetime(df2['date'], format='%Y%m%d')
        df2['precipitation'] = df2['precipitation'].shift(periods=1)

        df2['date_idx'] = df2['date']
        df2 = df2.set_index('date_idx')
        return df2

    def fillna_reindexed_nans(self, df):
        dt_reindex_nans = []
        [dt_reindex_nans.append(col) if df[col].isna().sum(
        ) < 10 and df[col].isna().sum() > 0 else None for col in df.columns]
        df.loc[:, dt_reindex_nans] = df.loc[:, dt_reindex_nans].ffill()
        return df

    def fillna_assume_zero(self, df, cols):
        df[cols] = df[cols].fillna(0)
        return df

    def fillna_mean(self, df, cols):
        for col in cols:
            df[col] = df.groupby(['month'])[col].transform(
                lambda x: x.fillna(x.mean()))
        df[cols] = df[cols].fillna(df[cols].mean())
        return df

    def fillna_bfill(self, df, cols):
        df[cols] = df[cols].bfill()
        df[cols] = df[cols].ffill()
        return df

    # def split_data(self, df):
    #     self.logger.info('Splitting data')
    #     df_size = len(df)
    #     train_size = int(0.8 * df_size)
    #     test_size = df_size - train_size

    #     train_dataset, test_dataset = random_split(df, [train_size, test_size])
    #     return train_dataset, test_dataset

    def split_data(self, df, input_variable):
        self.logger.info('Splitting data')
        train_dataset, test_dataset = train_test_split(
            df[input_variable], test_size=0.2, shuffle=False)
        return train_dataset, test_dataset  # X_train, X_test

    def initial_process(self, df1, df2):
        df1, df2 = pressure_to_kPa(df1, df2)

        df1 = self.process_dt_df1(df1)
        df2 = self.process_dt_df2(df2)

        to_impute = self.config['processing']['to_impute']
        impute_with = self.config['processing']['impute_with']
        df = self.fillna_from_df2(df1, df2, to_impute, impute_with)

        df = self.drop_cols(df)
        df = self.fillna_reindexed_nans(df)
        return df

    def further_process(self, df, input_variable):
        impute_zero_cols = self.config['processing']['impute_zero_cols']
        impute_mean_cols = self.config['processing']['impute_mean_cols']
        impute_bfill_cols = self.config['processing']['impute_bfill_cols']

        self.fillna_assume_zero(df, impute_zero_cols)
        self.fillna_mean(df, impute_mean_cols)
        self.fillna_bfill(df, impute_bfill_cols)

        self.split_data(df, input_variable)
        return df
