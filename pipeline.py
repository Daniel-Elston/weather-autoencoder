from __future__ import annotations

import logging
import warnings

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from src.data.load_data import RawDataLoader
from src.data.make_dataset import WeatherDataset
from src.data.processing import Processor
from src.data.transforms import StandardScaler
from src.data.transforms import ToTensor
from src.data.transforms import Windowing
from src.features.build_features import BuildFeatures
from utils.file_load import FileLoader
from utils.file_save import FileSaver
from utils.my_utils import dataset_stats
from utils.os_view import OSView
from utils.setup_env import setup_project_env
# import logging.config
# import numpy as np
# from src.data.transforms import Differencing
# from src.data.transforms import MinMaxScaler
warnings.filterwarnings("ignore")


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.data_paths = config['data_paths']
        self.input_var = self.config['input_variable']
        self.raw_loader = RawDataLoader(self.config)
        self.processor = Processor(self.config)
        self.feature_builder = BuildFeatures(self.config)
        self.loader = FileLoader()
        self.saver = FileSaver()
        self.logger = logging.getLogger(self.__class__.__name__)

    def run_load_data(self):
        self.raw_loader.load_data()
        df1, df2, df3, df4 = self.raw_loader.get_data()
        # self.raw_loader.get_data_info(df1, info='shape')
        return df1, df2

    def run_process_data(self, df1, df2, save=False):
        df = self.processor.initial_process(df1, df2)
        df = self.feature_builder.build_features(df)
        df = self.processor.further_process(df)
        if save:
            self.saver.save_file(df, self.config['processed_data'])
        train_df, test_df = self.processor.split_data(df, self.input_var)
        return train_df, test_df

    def main(self):
        self.logger.info(
            'Running pipeline ------------------------------------------------------------'
        )
        df1, df2 = self.run_load_data()
        train_df, test_df = self.run_process_data(df1, df2, save=True)
        means, stds, mins, maxs = dataset_stats(train_df)
        self.logger.info(
            'Creating datasets/dataloaders ------------------------------------------------'
        )
        transform = Compose([
            Windowing(window_size=14),
            # Differencing(),
            StandardScaler(means, stds),
            ToTensor(),
        ])

        train_dataset = WeatherDataset(
            series=train_df, window_size=14, transform=transform)
        test_dataset = WeatherDataset(
            series=test_df, window_size=14, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        OSView().get_visual(train_df, train_loader)
        OSView().get_visual(test_df, test_loader)
        self.logger.info(
            'Finished pipeline -----------------------------------------------------------'
        )

    def test(self):
        print('Testing pipeline')


if __name__ == '__main__':
    project_dir, config, set_log = setup_project_env()
    pipeline = DataPipeline(config)
    pipeline.main()
    # pipeline.test()
