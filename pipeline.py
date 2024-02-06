from __future__ import annotations

import logging.config
import warnings
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from src.data.load_data import RawDataLoader
from src.data.make_dataset import WeatherDataset
from src.data.processing import Processor
from src.data.transforms import StandardScaler
from src.data.transforms import ToTensor
from src.features.build_features import BuildFeatures
from utils.file_load import FileLoader
from utils.file_save import FileSaver
from utils.my_utils import dataset_stats
from utils.setup_env import setup_project_env
from utils.setup_logging import setup_logging
# import numpy as np
# from src.data.transforms import Differencing
# from src.data.transforms import MinMaxScaler
# from src.data.transforms import Windowing
# from utils.file_log import Logger
warnings.filterwarnings("ignore")


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.data_paths = config['data_paths']
        self.raw_loader = RawDataLoader(self.config)
        self.processor = Processor(self.config)
        self.feature_builder = BuildFeatures(self.config)
        self.loader = FileLoader()
        self.saver = FileSaver()
        # self.logger = Logger(
        #     'PipelineLog', f'{Path(__file__).stem}.log').get_logger()
        self.logger = logging.getLogger(__name__)

    def run_load_data(self):
        # self.logger.info('Loading data...')
        self.raw_loader.load_data()
        df1, df2, df3, df4 = self.raw_loader.get_data()
        self.raw_loader.get_data_info(df1, info='shape')
        # self.logger.info('Data loaded')
        return df1, df2

    def run_initial_process(self, df1, df2):
        # self.logger.info('Initial processing')
        df = self.processor.initial_process(df1, df2)
        return df

    def run_build_features(self, df):
        # self.logger.info('Building features')
        df = self.feature_builder.build_dt_features(df)
        return df

    def run_further_process(self, df):
        # self.logger.info('Further processing')
        df = self.processor.further_process(df)
        return df

    def split_data(self, df, input_variable):
        # self.logger.info('Splitting data')
        train, test = self.processor.split_data(df, input_variable)
        return train, test

    def run_save_data(self, df, path):
        # self.logger.info('Saving data...')
        self.saver.save_file(df, path)
        # self.logger.info(f'Data saved to {path}')

    def main(self):
        # self.logger.info('Running pipeline')
        # df1, df2 = self.run_load_data()
        # print(df1.head())

        # df = self.run_initial_process(df1, df2)
        # df = self.run_build_features(df)
        # df = self.run_further_process(df)

        # self.run_save_data(df, self.config['processed_data'])
        df = self.loader.load_file(self.config['processed_data'])

        input_variable = self.config['input_variable']
        train_df, test_df = self.split_data(df, input_variable)
        print(train_df)

        means, stds, mins, maxs = dataset_stats(train_df)

        transform = Compose([
            # Windowing(window_size=5),
            # Differencing(),
            StandardScaler(means, stds),
            ToTensor(),  # Convert dataframes/ndarrays to PyTorch tensors
        ])

        train_dataset = WeatherDataset(dataframe=train_df, transform=transform)
        # test_dataset = WeatherDataset(dataframe=test_df, transform=transform)

        # Visualize the first few transformed samples from the dataset
        for i in range(5):  # Adjust the range as needed
            sample = train_dataset[i]
            print(f"Transformed sample {i}:", sample)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Example to print the first batch from the train_loader
        for batch in train_loader:
            print("First batch from DataLoader:", batch)
            print("Batch shape:", batch.shape)
            break  # Only show the first batch

        # self.logger.info('Finished pipeline')

    def test(self):

        self.logger.debug(
            'This debug message goes to the dynamically named file')
        self.logger.info(
            'This info message goes to the dynamically named file')


if __name__ == '__main__':
    project_dir, config = setup_project_env()
    setup_logging(project_dir, f'{Path(__file__).stem}.log')

    pipeline = DataPipeline(config)
    pipeline.main()
    # pipeline.test()
