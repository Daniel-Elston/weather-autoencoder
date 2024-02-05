from __future__ import annotations

import warnings
from pathlib import Path

from src.data.load_data import DataLoader
from src.data.processing import Processor
from src.features.build_features import BuildFeatures
from utils.file_log import Logger
from utils.file_save import FileSaver
from utils.setup_env import setup_project_env
warnings.filterwarnings("ignore")


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.data_paths = config['data_paths']
        self.loader = DataLoader(self.config)
        self.processor = Processor(self.config)
        self.feature_builder = BuildFeatures(self.config)
        self.saver = FileSaver()
        self.logger = Logger(
            'PipelineLog', f'{Path(__file__).stem}.log').get_logger()

    def run_load_data(self):
        self.logger.info('Loading data...')
        self.loader.load_data()
        df1, df2, df3, df4 = self.loader.get_data()
        self.loader.get_data_info(df1, info='shape')
        self.logger.info('Data loaded')
        return df1, df2

    def run_initial_process(self, df1, df2):
        self.logger.info('Initial processing')
        df = self.processor.initial_process(df1, df2)
        return df

    def run_build_features(self, df):
        self.logger.info('Building features')
        df = self.feature_builder.build_dt_features(df)
        return df

    def run_further_process(self, df):
        self.logger.info('Further processing')
        df = self.processor.further_process(df)
        return df

    def split_data(self, df):
        self.logger.info('Splitting data')
        train, test = self.processor.split_data(df)
        return train, test

    def run_save_data(self, df, path):
        self.logger.info('Saving data...')
        self.saver.save_file(df, path)
        self.logger.info(f'Data saved to {path}')

    def main(self):
        self.logger.info('Running pipeline')
        df1, df2 = self.run_load_data()

        df = self.run_initial_process(df1, df2)
        df = self.run_build_features(df)
        df = self.run_further_process(df)
        train_df, test_df = self.split_data(df)

        self.run_save_data(df, self.config['processed_data'])

        self.logger.info('Finished pipeline')


if __name__ == '__main__':
    project_dir, config = setup_project_env()
    pipeline = DataPipeline(config)
    pipeline.main()
