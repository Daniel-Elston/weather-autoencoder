from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.pipeline import DataPipeline

warnings.filterwarnings("ignore")


class TestPipeline(DataPipeline):
    def __init__(self, config):
        super().__init__(config)
        self.train_path = Path('data/interim/art_daily_small_noise.csv')
        self.test_path = Path('data/interim/art_daily_jumpsup.csv')
        self.batch_size = 288

    def run_load_data(self):
        df1 = pd.read_csv(self.train_path)
        df2 = pd.read_csv(self.test_path)
        return df1, df2

    def run_process_data(self, df1, df2, save=False):
        df = pd.concat([df1, df2], ignore_index=False)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        train_df, test_df = train_test_split(
            df['value'], test_size=0.2, shuffle=False)
        train_df, val_df = train_test_split(
            train_df, test_size=0.25, shuffle=False)
        return train_df, val_df, test_df

    def main(self):
        self.logger.info(
            '============================== RUNNING TEST PIPELINE ==============================')
        super().main()
        self.logger.info(
            '============================== TEST PIPELINE COMPLETE ==============================')
