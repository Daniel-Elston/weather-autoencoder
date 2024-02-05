from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.load_data import DataLoader
from utils.file_log import Logger
from utils.setup_env import setup_project_env

project_dir, config = setup_project_env()


class BuildFeatures(DataLoader):
    def __init__(self, config):
        super().__init__(config)
        self.logger = Logger(
            'BuildFeaturesLog', f'{Path(__file__).stem}.log').get_logger()
        self.config = config
        self.variables = config['variables']
        self.windows = config['windows']

    def build_dt_features(self, df):
        self.logger.info('Building datetime features')
        df['year'] = pd.to_datetime(df['date']).dt.year
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['day'] = pd.to_datetime(df['date']).dt.day

        df['day_of_year'] = df['date'].dt.dayofyear
        df['month_year'] = df['date'].dt.to_period('M')
        df['season_num'] = df['season'].map(
            {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Autumn': 4})
        df['season_year'] = df['year'].astype(
            str)+'-'+df['season_num'].astype(str)
        df['season_start_month'] = df['season_num'].map(
            {1: '12', 2: '03', 3: '06', 4: '09'})
        df['season_dt'] = pd.to_datetime(df['year'].astype(
            str)+'-'+df['season_start_month'].astype(str)+'-01')
        return df

    def build_features(self, df):
        return self.build_dt_features(df)
