from __future__ import annotations

from src.data.base_processing import ProcessData
from utils.setup_env import setup_project_env
project_dir, config = setup_project_env()


class MakeDataset(ProcessData):
    def __init__(self, config):
        super().__init__(config)

    def get_data(self):
        self.load_data()

        # super().get_data()
        # self.get_data_info()
        return super().get_data()
