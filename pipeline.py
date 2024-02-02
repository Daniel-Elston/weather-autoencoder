from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from src.data.make_dataset import MakeDataset
from utils.file_log import Logger
from utils.setup_env import setup_project_env
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.logger = Logger(
            'PipelineLog', f'{Path(__file__).stem}.log').get_logger()
        self.data_path_1 = config['data_path_1']
        self.data_path_2 = config['data_path_2']

    def main(self):
        self.logger.info('Running pipeline')

        make_dataset = MakeDataset(self.config)
        df1, df2 = make_dataset.get_data()
        make_dataset.get_data_info(df1)

        self.logger.info('Finished pipeline')


if __name__ == '__main__':
    project_dir, config = setup_project_env()
    pipeline = DataPipeline(config)
    pipeline.main()
