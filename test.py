from __future__ import annotations

import logging
import warnings

import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from src.data.load_data import RawDataLoader
from src.data.make_dataset import WeatherDataset
from src.data.processing import Processor
from src.data.transforms import MinMaxScaler
from src.data.transforms import ToTensor
from src.data.transforms import Windowing
from src.features.build_features import BuildFeatures
from src.models.model_params import ModelParams
from src.models.predict_model import predict
from src.models.train_model import train_model
from src.models.uae import ConvAutoencoder
from src.visualization.results_visuals import plot_losses
from src.visualization.results_visuals import plot_preds
from src.visualization.results_visuals import plot_reconstructed
from utils.file_load import FileLoader
from utils.file_save import FileSaver
from utils.my_utils import dataset_stats
from utils.setup_env import setup_project_env
# from src.models.uae import LinAutoencoder
# from src.data.transforms import StandardScaler
# from utils.os_view import OSView
warnings.filterwarnings("ignore")


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.data_paths = config['data_paths']
        self.input_var = self.config['input_variable']
        self.window_size = self.config['window_size']
        self.batch_size = self.config['batch_size']

        self.raw_loader = RawDataLoader(self.config)
        self.processor = Processor(self.config)
        self.feature_builder = BuildFeatures(self.config)
        self.loader = FileLoader()
        self.saver = FileSaver()
        self.params = ModelParams()
        self.logger = logging.getLogger(self.__class__.__name__)


def test(self):
    print('Testing pipeline')

    train_df = pd.read_csv('data/interim/art_daily_small_noise.csv')
    test_df = pd.read_csv('data/interim/art_daily_jumpsup.csv')

    def process_data(df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df.drop('Unnamed: 0', axis=1)
        df = df['value']
        return df
    train_df = process_data(train_df)
    test_df = process_data(test_df)

    means, stds, mins, maxs = dataset_stats(train_df)

    transform = Compose([
        Windowing(window_size=288),
        # StandardScaler(means, stds),
        MinMaxScaler(mins, maxs),
        ToTensor(),
    ])

    train_dataset = WeatherDataset(
        series=train_df, window_size=288, transform=transform)
    test_dataset = WeatherDataset(
        series=test_df, window_size=288, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False)

    # model = LinAutoencoder(
    #     input_dim=288, latent_dims=128)
    model = ConvAutoencoder()

    train_loss, val_loss = train_model(
        model, train_loader, test_loader, self.params)
    plot_reconstructed(
        model, test_loader, device=self.params.device, sample_size=1)

    # scaler = StandardScaler(means, stds)
    scaler = MinMaxScaler(mins, maxs)
    originals, predictions = predict(model, test_loader, scaler)

    plot_preds(originals, predictions)
    plot_losses(train_loss, val_loss)


if __name__ == '__main__':
    project_dir, config, set_log = setup_project_env()
    pipeline = DataPipeline(config)
    pipeline.test()
