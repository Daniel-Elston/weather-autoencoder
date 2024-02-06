from __future__ import annotations

import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.file_load import FileLoader
from utils.setup_env import setup_project_env
# import time


class Converter:
    def __init__(self, path, config):
        self.path = path
        self.config = config
        self.loader = FileLoader()
        self.logger = logging.getLogger(self.__class__.__name__)

    def convert_to_parq(self):
        df = pd.read_excel(self.path)
        parq_file = self.path.replace('.xlsx', '.parq')
        pq.write_table(pa.Table.from_pandas(
            df), parq_file, compression='snappy')
        return parq_file

    def run(self):
        self.convert_to_parq()

    def test(self):
        # t0 = time.time()
        # df = self.loader.load_file(self.path)
        # df = self.loader.load_file(self.path)
        # t1 = time.time()
        # total = (t1-t0)/2
        # print(f"Time to load data: {total:.2f} seconds")
        pass


if __name__ == '__main__':
    project_dir, config, setup_logs = setup_project_env()
    path = 'C:/Users/delst/workspace/weather-autoencoder/data/processed/processed_data.xlsx'
    convert = Converter(path, config)
    convert.run()
    # convert.test()
