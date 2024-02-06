from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.file_load import FileLoader
from utils.file_log import Logger
from utils.my_utils import str_contains
from utils.setup_env import setup_project_env

project_dir, config = setup_project_env()


class RawDataLoader:
    def __init__(self, config):
        self.config = config
        self.loader = FileLoader()
        self.logger = Logger(
            'RawDataLoaderLog', f'{Path(__file__).stem}.log').get_logger()
        self.data_paths = config['data_paths']
        self.data = []

    def load_data(self):
        self.load_data_chunks()
        self.load_files()

    def get_data(self):
        self.df1 = pd.concat(self.data)
        return self.df1, self.df2, self.df3, self.df4

    def load_data_chunks(self):
        self.logger.info(f'Loading dataset 1: {self.data_paths[0]}')
        try:
            daily_weather = pq.ParquetFile(self.data_paths[0])

            chunk_size = 2E4
            self.data = []
            batch_num = 0
            n_rows = 0

            for batch in daily_weather.iter_batches(batch_size=int(chunk_size)):
                chunk_df = pa.Table.from_batches([batch]).to_pandas(
                    split_blocks=True, self_destruct=True)
                df_city = str_contains(chunk_df, 'city_name', 'London')
                if df_city.empty:
                    continue
                self.data.append(df_city)
                batch_num += 1
                n_rows += df_city.shape[0]

                self.logger.info(
                    f'Batch {batch_num} loaded, shape: {df_city.shape}')
        except Exception as e:
            self.logger.error(f'Error loading data: {e}')
            raise e
        self.logger.info(f'Dataset 0 loaded, N_rows: {n_rows}')

    def load_files(self):
        try:
            frame_store = []
            for i, path in enumerate(self.data_paths[1:], start=1):
                self.logger.info(f'Loading dataset {i}: {path}')
                df = self.loader.load_file(path)
                frame_store.append(df)
                self.logger.info(f'Dataset {i} loaded, shape: {df.shape}')
        except Exception as e:
            self.logger.error(f'Error loading data: {e}')
            raise e
        self.df2, self.df3, self.df4 = frame_store

    def get_data_info(self, df, info):
        try:
            if info == 'head':
                self.logger.info(f'Data Head: {df.head()}')
            elif info == 'info':
                self.logger.info(f'Data Info: {df.info()}')
            elif info == 'describe':
                self.logger.info(f'Data Description: {df.describe()}')
            elif info == 'shape':
                self.logger.info(f'Data Shape: {df.shape}')
            elif info == 'columns':
                self.logger.info(f'Data Columns: {df.columns}')
            elif info == 'dtypes':
                self.logger.info(f'Data Types: {df.dtypes}')
            elif info == 'nans':
                self.logger.info(f'Data Nans: {df.isna().sum()}')
            elif info == 'len':
                self.logger.info(f'Data Length: {len(df)}')
            else:
                self.logger.info(f'Data Head: {df.head()}')
                self.logger.info(f'Data Info: {df.info()}')
                self.logger.info(f'Data Description: {df.describe()}')
                self.logger.info(f'Data Shape: {df.shape}')
                self.logger.info(f'Data Columns: {df.columns}')
                self.logger.info(f'Data Types: {df.dtypes}')
                self.logger.info(f'Data Nans: {df.isna().sum()}')
                self.logger.info(f'Data Length: {len(df)}')
        except Exception as e:
            self.logger.error(f'Error getting data info: {e}')
