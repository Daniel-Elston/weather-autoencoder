from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.file_load import FileLoader
from utils.file_log import Logger
from utils.setup_env import setup_project_env
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

project_dir, config = setup_project_env()


class ProcessData:
    def __init__(self, config):
        self.config = config
        self.loader = FileLoader()
        self.logger = Logger(
            'BaseProcessLog', f'{Path(__file__).stem}.log').get_logger()
        self.data_path_1 = config['data_path_1']
        self.data_path_2 = config['data_path_2']
        self.data = None
        self.df1 = None
        self.df2 = None

    def str_contains(self, df, column_name, value):
        """Filter DataFrame rows where `column_name` contains `value`."""
        return df[df[column_name].str.contains(value, na=False)]

    def load_data(self):
        self.logger.info(f'Loading dataset 1: {self.data_path_1}')
        try:
            daily_weather = pq.ParquetFile(self.data_path_1)

            chunk_size = 2E4
            self.data = []
            batch_num = 0
            total_batch_shape = 0

            for batch in daily_weather.iter_batches(batch_size=int(chunk_size)):
                chunk_df = pa.Table.from_batches([batch]).to_pandas(
                    split_blocks=True, self_destruct=True)
                df_city = self.str_contains(chunk_df, 'city_name', 'London')
                if df_city.empty:
                    continue
                self.data.append(df_city)
                total_batch_shape += df_city.shape[0]
                batch_num += 1

                self.logger.info(
                    f'Batch {batch_num} loaded, shape: {df_city.shape}')
        except Exception as e:
            self.logger.error(f'Error loading data: {e}')
            raise e

        self.logger.info(f'Dataset 1 loaded, N_rows: {total_batch_shape}')

        self.logger.info(f'Loading dataset 2: {self.data_path_2}')
        try:
            self.df2 = self.loader.load_file(self.data_path_2)
        except Exception as e:
            self.logger.error(f'Error loading data: {e}')
            raise e
        self.logger.info(f'Dataset 2 loaded, shape: {self.df2.shape}')

    def get_data(self):
        self.df1 = pd.concat(self.data)
        self.df2 = self.loader.load_file(self.data_path_2)
        return self.df1, self.df2

    def get_data_info(self, df):
        try:
            # self.logger.info(f'Data Head: {df.head()}')
            # self.logger.info(f'Data Info: {df.info()}')
            # self.logger.info(f'Data Description: {df.describe()}')
            self.logger.info(f'Data Shape: {df.shape}')
        except Exception as e:
            self.logger.error(f'Error getting data info: {e}')
