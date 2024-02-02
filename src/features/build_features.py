from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.load_data import DataLoader
from utils.file_log import Logger
from utils.setup_env import setup_project_env

project_dir, config = setup_project_env()


class BuildFeatures(DataLoader):
    def __init__(self, config, df):
        super().__init__(config)
        self.logger = Logger(
            'BuildFeaturesLog', f'{Path(__file__).stem}.log').get_logger()
        self.df = df
        self.config = config
        self.variables = config['variables']
        self.windows = config['windows']

    def build_dt_features(self):
        self.df['year'] = pd.to_datetime(self.df['date']).dt.year
        self.df['month'] = pd.to_datetime(self.df['date']).dt.month
        self.df['day'] = pd.to_datetime(self.df['date']).dt.day

        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        self.df['month_year'] = self.df['date'].dt.to_period('M')
        self.df['season_num'] = self.df['season'].map(
            {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Autumn': 4})
        self.df['season_year'] = self.df['year'].astype(
            str)+'-'+self.df['season_num'].astype(str)
        self.df['season_start_month'] = self.df['season_num'].map(
            {1: '12', 2: '03', 3: '06', 4: '09'})
        self.df['season_dt'] = pd.to_datetime(self.df['year'].astype(
            str)+'-'+self.df['season_start_month'].astype(str)+'-01')
        return self.df
