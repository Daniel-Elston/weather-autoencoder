from __future__ import annotations

import logging
import warnings
# import time

# from torch.utils.data import DataLoader
# from torchvision.transforms import Compose

# from src.data.load_data import RawDataLoader
# from src.data.make_dataset import WeatherDataset
# from src.data.processing import Processor
# from src.data.transforms import StandardScaler
# from src.data.transforms import ToTensor
# from src.features.build_features import BuildFeatures
# from utils.file_load import FileLoader
# from utils.file_save import FileSaver
# from utils.my_utils import dataset_stats
# from utils.setup_env import setup_project_env
# import logging.config
# import numpy as np
# from src.data.transforms import Differencing
# from src.data.transforms import MinMaxScaler
# from src.data.transforms import Windowing
warnings.filterwarnings("ignore")


class OSView:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_visual(self, dataset, loader):
        self.logger.warning(
            '--------------------------------------------------------------------------------------------')
        print('----------------------------------------------')
        for i in range(5):
            sample = dataset[i]
            print(f"Transformed sample {i}:\n", sample)

        for batch in loader:
            print("First batch from DataLoader:\n", batch)
            print("Batch shape:\n", batch.shape)
            break
        self.logger.warning(
            '--------------------------------------------------------------------------------------------')
