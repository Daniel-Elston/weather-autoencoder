from __future__ import annotations

from pathlib import Path

from src.data.load_data import DataLoader
from utils.file_log import Logger
from utils.setup_env import setup_project_env
project_dir, config = setup_project_env()


class MakeDataset(DataLoader):
    def __init__(self, config):
        super().__init__(config)
        self.logger = Logger(
            'MakeDatasetLog', f'{Path(__file__).stem}.log').get_logger()
