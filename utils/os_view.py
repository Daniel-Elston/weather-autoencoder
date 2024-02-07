from __future__ import annotations

import logging
import warnings
warnings.filterwarnings("ignore")


class OSView:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_visual(self, dataset, loader):
        self.logger.warning(
            '--------------------------------------------------------------------------------------------')
        print('----------------------------------------------')
        for i in range(5):
            sample = dataset[i]
            print(f"Transformed sample {i}:\n", sample)

        for batch in loader:
            print("First batch from DataLoader:\n", batch)
            print("Batch shape:\n", batch.shape)
            break
        self.logger.warning(
            '--------------------------------------------------------------------------------------------')
