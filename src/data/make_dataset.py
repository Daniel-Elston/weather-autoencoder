from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from utils.file_log import Logger
from utils.setup_env import setup_project_env
# import torch
project_dir, config = setup_project_env()


class WeatherDataset(Dataset):
    """Weather dataset"""

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        # self.variable = variable
        self.transform = transform
        self.logger = Logger(
            'MakeDatasetLog', f'{Path(__file__).stem}.log').get_logger()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = np.array([self.dataframe.iloc[idx]])
        if self.transform:
            sample = self.transform(sample)
        return sample
