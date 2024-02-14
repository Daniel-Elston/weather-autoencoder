from __future__ import annotations

import logging

import numpy as np
import torch

from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


class Windowing:
    """Generate windows of consecutive data points."""
    counter = 0

    def __init__(self, window_size):
        self.window_size = window_size
        self.n_logs = 1
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, data):

        if Windowing.counter < self.n_logs:
            self.logger.debug(
                f'Generating windows of size: {self.window_size}')

        windows = []
        for i in range(len(data) - self.window_size + 1):
            window = data[i:i + self.window_size]
            windows.append(window)
        Windowing.counter += 1
        return np.array(windows)


class Differencing:
    """Calculate the difference between consecutive data points."""
    counter = 0

    def __init__(self):
        self.n_logs = 1
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, data):
        if Differencing.counter < self.n_logs:
            self.logger.debug(
                f'Aplying differencing to sample: {type(data)}')
        Differencing.counter += 1
        return np.diff(data, n=1, axis=0)


class StandardScaler:
    """Normalize the sample"""
    counter = 0

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.n_logs = 1
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, tensor):
        if StandardScaler.counter < self.n_logs:
            self.logger.debug(
                f"Input Shape: {tensor.shape}, type: {type(tensor)}")
        tensor = (tensor - self.mean) / self.std
        if StandardScaler.counter < self.n_logs:
            self.logger.debug(
                f"Output Shape: {tensor.shape}, type: {type(tensor)}")
        StandardScaler.counter += 1
        return tensor

    def inverse_transform(self, tensor):
        return tensor * self.std + self.mean


class MinMaxScaler:
    """Scale the sample"""
    counter = 0

    def __init__(self, min_val, max_val, feature_range=(0, 1)):
        self.min_val = min_val
        self.max_val = max_val
        self.range = feature_range
        self.n_logs = 1
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, tensor):
        if MinMaxScaler.counter < self.n_logs:
            self.logger.debug(f'Scaling sample: {type(tensor)}')
        tensor = (tensor - self.min_val) / (self.max_val - self.min_val)
        min_range, max_range = self.range
        tensor = tensor * (max_range - min_range) + min_range
        MinMaxScaler.counter += 1
        return tensor

    def inverse_transform(self, tensor):
        tensor = (tensor - self.range[0]) / (self.range[1] - self.range[0])
        tensor = tensor * (self.max_val - self.min_val) + self.min_val
        return tensor


class ToTensor:
    counter = 0
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        self.n_logs = 1
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, sample):
        if ToTensor.counter < self.n_logs:
            self.logger.debug(f'Converting sample to tensor: {type(sample)}')
        tensor = torch.from_numpy(sample).float()
        if ToTensor.counter < self.n_logs:
            self.logger.debug(f'Sample converted to tensor: {type(tensor)}')
        ToTensor.counter += 1
        return tensor
