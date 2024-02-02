from __future__ import annotations

import warnings
from pathlib import Path

from utils.file_load import FileLoader
from utils.file_log import Logger
from utils.setup_env import setup_project_env
warnings.filterwarnings("ignore")


class AnalyseData:
    def __init__(self, config):
        self.config = config
        self.logger = Logger(
            'AnalyseDataLog', f'{Path(__file__).stem}.log').get_logger()

    def main(self):
        self.logger.info('Running analysis')
        file_loader = FileLoader()
        file_loader.load_file(self.config['processed_data'])

        # visulaise data from viz.py
        # save plots to file

        self.logger.info('Finished analysis')


if __name__ == '__main__':
    project_dir, config = setup_project_env()
    pipeline = AnalyseData(config)
    pipeline.main()
