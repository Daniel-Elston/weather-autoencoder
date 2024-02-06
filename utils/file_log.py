# from __future__ import annotations
# import logging
# import os
# from logging.handlers import RotatingFileHandler
# from utils.setup_env import setup_project_env
# project_dir, config = setup_project_env()
# class Logger:
#     def __init__(self, name, log_file):
#         """
#         Initialize the Logger.
#         :param name: Name of the logger.
#         :param log_file: File path for the log file.
#         """
#         # Create or get a logger
#         level = config['logging']['level']
#         logger = logging.getLogger(name)
#         # Check if the logger already has handlers to avoid adding them again
#         if not logger.handlers:
#             logger.setLevel(getattr(logging, level))
#             # Create handlers
#             file_handler = RotatingFileHandler(
#                 os.path.join(project_dir, f'log/{log_file}'), maxBytes=1000000, backupCount=5)
#             console_handler = logging.StreamHandler()
#             # Set levels
#             console_handler.setLevel(logging.INFO)
#             # Create formatters and add to handlers
#             file_formatter = logging.Formatter(
#                 '%(pathname)s - %(asctime)s - %(levelname)s - %(filename)s'
#                 ' - %(lineno)d - %(module)s - %(funcName)s - %(name)s - %(message)s')
#             console_formatter = logging.Formatter(
#                 '%(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s')
#             file_handler.setFormatter(file_formatter)
#             console_handler.setFormatter(console_formatter)
#             # Add handlers to the logger
#             logger.addHandler(file_handler)
#             logger.addHandler(console_handler)
#         self.logger = logger
#     def get_logger(self):
#         """
#         Returns the configured logger.
#         """
#         return self.logger
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from utils.setup_env import setup_project_env

project_dir, config = setup_project_env()


class Logger:
    def __init__(self, name, log_file):
        """
        Initialize the Logger.

        :param name: Name of the logger.
        :param log_file: File path for the log file.
        """
        file_level = config['logging']['file_level']  # Expecting 'DEBUG', 'INFO', etc.
        console_level = config['logging']['console_level']

        logger = logging.getLogger(name)
        if not logger.handlers:
            # logger.setLevel(logging.DEBUG)
            logger.setLevel(file_level)

            # Create handlers with a common formatter
            file_handler = RotatingFileHandler(
                os.path.join(project_dir, f'log/{log_file}'), maxBytes=1000000, backupCount=5)
            console_handler = logging.StreamHandler()

            file_formatter = logging.Formatter(
                '%(pathname)s - %(asctime)s - %(levelname)s - %(filename)s'
                ' - %(lineno)d - %(module)s - %(funcName)s - %(name)s - %(message)s')
            console_formatter = logging.Formatter(
                '%(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s')

            file_handler.setFormatter(file_formatter)
            console_handler.setFormatter(console_formatter)

            # Set levels for each handler based on config
            file_handler.setLevel(getattr(logging, file_level, "INFO"))
            console_handler.setLevel(getattr(logging, console_level, "INFO"))

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        self.logger = logger

    def get_logger(self):
        """
        Returns the configured logger.
        """
        return self.logger
