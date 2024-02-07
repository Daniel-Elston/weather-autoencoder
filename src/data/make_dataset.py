from __future__ import annotations

import logging

from torch.utils.data import Dataset

from utils.setup_env import setup_project_env
# import torch
project_dir, config, setup_logs = setup_project_env()


class WeatherDataset(Dataset):
    """Weather dataset (univariate)
    Args:
        series (pd.Series): Time series data
        transform (callable, optional): Optional transform to be applied
            on a sample.
    Returns:
        sample (np.array): Sample data
    """

    def __init__(self, series, window_size, transform=None):
        self.series = series
        self.window_size = window_size
        self.transform = transform
        self.logger = logging.getLogger(self.__class__.__name__)
        # self.logger.debug('Initialized WeatherDataset with %d samples', len(self.series))
        self.logger.debug(
            f"Initialized WeatherDataset. Shape: {self.series.shape}, type: {type(self.series)}")

    def __len__(self):
        # Adjust length to account for windowing
        return len(self.series) - self.window_size + 1

    def __getitem__(self, idx):
        # Generate a window starting at index idx
        window = self.series.iloc[idx:idx + self.window_size].values
        if self.transform:
            window = self.transform(window)
        if idx < 3:
            self.logger.debug(
                f'Window Num {idx}, Shape: {window.shape}, type: {type(window)}')
        return window
