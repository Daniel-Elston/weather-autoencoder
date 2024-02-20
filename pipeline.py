from __future__ import annotations

import logging
import warnings
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from src.data.load_data import RawDataLoader
from src.data.make_dataset import WeatherDataset
from src.data.processing import Processor
from src.data.transforms import MinMaxScaler
from src.data.transforms import ToTensor
from src.data.transforms import Windowing
from src.features.build_features import BuildFeatures
from src.models.evaluate_model import evaluations
from src.models.model_params import ModelParams
from src.models.predict_model import predict
from src.models.train_model import train_model
from src.models.uae import ConvAutoencoder
from src.visualization.results_visuals import ResultsVisuals
from utils.file_load import FileLoader
from utils.file_save import FileSaver
from utils.my_utils import dataset_stats
from utils.my_utils import save_model_summary
from utils.setup_env import setup_project_env
warnings.filterwarnings("ignore")


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.data_paths = config['data_paths']
        self.input_var = config['input_variable']
        self.window_size = config['window_size']
        self.batch_size = config['batch_size']

        self.raw_loader = RawDataLoader(config)
        self.processor = Processor(config)
        self.feature_builder = BuildFeatures(config)
        self.loader = FileLoader()
        self.saver = FileSaver()
        self.params = ModelParams()
        self.logger = logging.getLogger(self.__class__.__name__)

    def run_load_data(self):
        self.raw_loader.load_data()
        df1, df2, _, _ = self.raw_loader.get_data()
        return df1, df2

    def run_process_data(self, df1, df2, save=False):
        df = self.processor.initial_process(df1, df2)
        df = self.feature_builder.build_features(df)
        df = self.processor.further_process(df)
        if save:
            self.saver.save_file(df, Path(self.config['processed_data']))
        train_df, val_df, test_df = self.processor.split_data(
            df, self.input_var)
        return train_df, val_df, test_df

    def create_loader(self, series, window_size, batch_size, transform=None):
        dataset = WeatherDataset(
            series=series, window_size=window_size, transform=transform)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    def main(self):
        self.logger.info(
            'Running Pipeline ------------------------------------------------------------'
        )
        df1, df2 = self.run_load_data()
        train_df, val_df, test_df = self.run_process_data(df1, df2, save=False)
        means, stds, mins, maxs = dataset_stats(train_df)

        self.logger.info(
            'Creating Datasets/Dataloaders ------------------------------------------------'
        )

        transform = Compose([
            Windowing(window_size=self.window_size),
            MinMaxScaler(mins, maxs),
            ToTensor(),
        ])

        train_loader = self.create_loader(
            train_df, self.window_size, self.batch_size, transform=transform)
        val_loader = self.create_loader(
            val_df, self.window_size, self.batch_size, transform=transform)
        test_loader = self.create_loader(
            test_df, self.window_size, self.batch_size, transform=transform)

        self.logger.info(
            'Training Model ------------------------------------------------------------'
        )
        model = ConvAutoencoder()

        train_loss, val_loss = train_model(
            model, train_loader, val_loader, self.params)

        scaler = MinMaxScaler(mins, maxs)
        x_test, x_test_preds = predict(model, test_loader, scaler, self.params)

        self.logger.info(
            'Model Evaluation ----------------------------------------------------------'
        )
        save_model_summary(config, model)

        anomalies, test_mae_loss, _ = evaluations(
            x_test, x_test_preds)

        eval_plot = ResultsVisuals(x_test, x_test_preds)
        eval_plot.eval_plotting(train_loss, val_loss, test_mae_loss, anomalies)

        self.logger.info(
            'Pipeline Complete ------------------------------------------------'
        )


if __name__ == '__main__':
    project_dir, config, set_log = setup_project_env()
    pipeline = DataPipeline(config)
    pipeline.main()
