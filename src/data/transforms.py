from __future__ import annotations

import logging

import numpy as np
import torch

from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


class Windowing:
    """Generate windows of consecutive data points."""

    def __init__(self, window_size):
        self.window_size = window_size
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, data):
        self.logger.debug(f'Generating windows of size: {self.window_size}')
        windows = []
        for i in range(len(data) - self.window_size + 1):
            window = data[i:i + self.window_size]
            windows.append(window)
        print("Windowing applied:", windows)
        # self.logger.debug('Windowing applied: %s', windows[0])
        return np.array(windows)


class Differencing:
    """Calculate the difference between consecutive data points."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, data):
        self.logger.info(f'Aplying differencing to sample: {type(data)}')
        return np.diff(data, n=1, axis=0)


class StandardScaler:
    """Normalize the sample"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, tensor):
        self.logger.debug(f'Scaling sample: {type(tensor)}')
        return (tensor - self.mean) / self.std


class MinMaxScaler:
    """Scale the sample"""

    def __init__(self, min_val, max_val, feature_range=(0, 1)):
        self.min_val = min_val
        self.max_val = max_val
        self.range = feature_range
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, tensor):
        self.logger.debug(f'Scaling sample: {type(tensor)}')
        tensor = (tensor - self.min_val) / (self.max_val - self.min_val)
        min_range, max_range = self.range
        tensor = tensor * (max_range - min_range) + min_range
        return tensor


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, sample):
        # self.logger.debug(f'Converting sample to tensor: {type(sample)}')
        # self.logger.info(f'THIS IS INFO: {sample}')
        return torch.from_numpy(sample).float()
