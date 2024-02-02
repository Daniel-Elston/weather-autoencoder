from __future__ import annotations

import warnings
from pathlib import Path

from src.data.load_data import DataLoader
from src.data.processing import Processor
from src.features.build_features import BuildFeatures
from utils.file_log import Logger
from utils.file_save import FileSaver
from utils.my_utils import pressure_to_kPa
from utils.setup_env import setup_project_env
warnings.filterwarnings("ignore")


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.logger = Logger(
            'PipelineLog', f'{Path(__file__).stem}.log').get_logger()
        self.data_paths = config['data_paths']

    def process_data(self, df1, df2):
        self.logger.info('Processing data')
        df1, df2 = pressure_to_kPa(df1, df2)

        processor = Processor(self.config)
        df1 = processor.process_dt_df1(df1)
        df2 = processor.process_dt_df2(df2)

        to_impute = self.config['processing']['to_impute']
        impute_with = self.config['processing']['impute_with']
        df = processor.fillna_from_df2(df1, df2, to_impute, impute_with)

        df = processor.drop_cols(df)
        df = processor.fillna_reindexed_nans(df)
        return df

    def build_features(self, df):
        self.logger.info('Building features')
        fe = BuildFeatures(self.config, df)
        df = fe.build_dt_features()
        return df

    def further_process(self, df):
        self.logger.info('Further processing data')
        processor = Processor(self.config)

        impute_zero_cols = self.config['processing']['impute_zero_cols']
        impute_mean_cols = self.config['processing']['impute_mean_cols']
        impute_bfill_cols = self.config['processing']['impute_bfill_cols']

        processor.fillna_assume_zero(df, impute_zero_cols)
        processor.fillna_mean(df, impute_mean_cols)
        processor.fillna_bfill(df, impute_bfill_cols)
        return df

    def main(self):
        self.logger.info('Running pipeline')
        loader = DataLoader(self.config)

        loader.load_data()
        df1, df2, df3, df4 = loader.get_data()
        loader.get_data_info(df1, info='shape')

        df = self.process_data(df1, df2)
        df = self.build_features(df)
        df = self.further_process(df)
        loader.get_data_info(df, info='shape')

        self.logger.info(f'Saving processed data to {
                         self.config['processed_data']}')

        file_saver = FileSaver()
        file_saver.save_file(df, self.config['processed_data'])

        self.logger.info('Finished pipeline')


if __name__ == '__main__':
    project_dir, config = setup_project_env()
    pipeline = DataPipeline(config)
    pipeline.main()
