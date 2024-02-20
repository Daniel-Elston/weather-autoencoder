from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline import DataPipeline
from utils.setup_env import setup_project_env

warnings.filterwarnings("ignore")


class TestPipeline(DataPipeline):
    def __init__(self, config):
        super().__init__(config)
        self.train_path = Path('data/interim/art_daily_small_noise.csv')
        self.test_path = Path('data/interim/art_daily_jumpsup.csv')
        self.logger = logging.getLogger(self.__class__.__name__)

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


if __name__ == '__main__':
    project_dir, config, set_log = setup_project_env()
    test_pipeline = TestPipeline(config)
    test_pipeline.main()
