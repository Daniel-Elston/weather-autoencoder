from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from src.visualization.visualize import VisualiseByYear
from src.visualization.visualize import VisualiseFull
from src.visualization.visualize import VisualiseOther
from utils.file_load import FileLoader
from utils.file_log import Logger
from utils.setup_env import setup_project_env
warnings.filterwarnings("ignore")


class AnalyseData:
    def __init__(self, config, df):
        self.config = config
        self.loader = FileLoader()
        self.logger = Logger(
            'AnalyseDataLog', f'{Path(__file__).stem}.log').get_logger()
        self.df = None

    def plot_full_timescale(self):
        vis_full = VisualiseFull(self.config, self.df)
        vis_full.plot_var_full(
            time_scale='month',
            save=True)
        vis_full.plot_rolling_means(
            windows=[30, 90], time_scale='month',
            save=True)

    def plot_by_year(self):
        vis_by_year = VisualiseByYear(self.config, self.df)
        vis_by_year.plot_axis(
            time_scale='day',
            save=True)
        vis_by_year.plot_rolling_means(
            windows=[30, 90], time_scale='day',
            save=True)

    def plot_other(self):
        vis_other = VisualiseOther(self.config, self.df)
        means, stds, cvs = vis_other.periodic_stats(
            variable='avg_temp_c', freq='month', stat='cv',
            save=True)
        vis_other.plot_grouped_years(
            variable='avg_temp_c', agg_type='mean',
            group_1=np.arange(1973, 1978), group_2=np.arange(2017, 2022),
            save=True)
        vis_other.plot_decade_var(
            variable='precipitation_mm', plt_type='bar', inc_err=False,
            save=True)

    def main(self):
        self.logger.info('Running analysis')
        self.logger.info('Loading data...')
        self.df = self.loader.load_file(self.config['processed_data'])

        self.logger.info('Plotting data')
        self.plot_full_timescale()
        self.plot_by_year()
        self.plot_other()

        self.logger.info(f'Plots saved to file path: {
                         self.config["fig_path"]}')
        self.logger.info('Finished analysis')


if __name__ == '__main__':
    project_dir, config = setup_project_env()
    pipeline = AnalyseData(config, None)
    pipeline.main()
