from __future__ import annotations

import logging
import warnings

from torch.utils.data import DataLoader
from torchinfo import summary
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
from src.visualization.results_visuals import plot_anomalies
from src.visualization.results_visuals import plot_losses
from src.visualization.results_visuals import plot_mae_loss
from src.visualization.results_visuals import plot_preds
from utils.file_load import FileLoader
from utils.file_save import FileSaver
from utils.my_utils import dataset_stats
from utils.setup_env import setup_project_env
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

    def run_load_data(self):
        self.raw_loader.load_data()
        df1, df2, df3, df4 = self.raw_loader.get_data()
        return df1, df2

    def run_process_data(self, df1, df2, save=False):
        df = self.processor.initial_process(df1, df2)
        df = self.feature_builder.build_features(df)
        df = self.processor.further_process(df)
        if save:
            self.saver.save_file(df, self.config['processed_data'])
        train_df, val_df, test_df = self.processor.split_data(
            df, self.input_var)
        return train_df, val_df, test_df

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

        train_dataset = WeatherDataset(
            series=train_df, window_size=self.window_size, transform=transform)
        val_dataset = WeatherDataset(
            series=val_df, window_size=self.window_size, transform=transform)
        test_dataset = WeatherDataset(
            series=test_df, window_size=self.window_size, transform=transform)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False)

        self.logger.info(
            'Training Model ------------------------------------------------------------'
        )
        model = ConvAutoencoder()

        train_loss, val_loss = train_model(
            model, train_loader, val_loader, self.params)

        scaler = MinMaxScaler(mins, maxs)
        x_train, x_train_preds = predict(model, train_loader, scaler)
        x_test, x_test_preds = predict(model, test_loader, scaler)

        self.logger.info(
            'Model Evaluation ----------------------------------------------------------'
        )
        with open(self.config['summary_path'], 'w', encoding='utf-8') as f:
            print(summary(model, (1, 1, self.window_size)), file=f)

        anomalies, test_mae_loss, test_errors = evaluations(
            x_test, x_test_preds)

        plot_preds(x_test, x_test_preds)
        plot_losses(train_loss, val_loss)
        plot_mae_loss(test_mae_loss)
        plot_anomalies(x_test[:, 0], x_test_preds[:, 0], anomalies[:, 0])

        self.logger.info(
            'Pipeline Complete ------------------------------------------------'
        )


if __name__ == '__main__':
    project_dir, config, set_log = setup_project_env()
    pipeline = DataPipeline(config)
    pipeline.main()
