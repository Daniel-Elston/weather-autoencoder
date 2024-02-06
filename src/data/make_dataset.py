from __future__ import annotations

import logging

import numpy as np
from torch.utils.data import Dataset

from utils.setup_env import setup_project_env
# import torch
project_dir, config, setup_logs = setup_project_env()


class WeatherDataset(Dataset):
    """Weather dataset"""

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        # self.variable = variable
        self.transform = transform
        self.logger = logging.getLogger(self.__class__.__name__)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = np.array([self.dataframe.iloc[idx]])
        if self.transform:
            sample = self.transform(sample)
        self.logger.debug(f'Index: {idx}, Sample: {sample}')
        return sample
