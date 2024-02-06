from __future__ import annotations

import logging.config
from pathlib import Path


def setup_logging(
        project_dir, log_file_name,
        file_level='DEBUG', console_level='DEBUG'):
    """Setup logging configuration with dynamic log file naming and levels."""

    log_file_path = Path(project_dir, 'log', log_file_name)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'detailed': {
                'format': '%(pathname)s - %(asctime)s - %(levelname)s - %(filename)s'
                          ' - %(lineno)d - %(module)s - %(funcName)s - %(name)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'simple': {
                'format': '%(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': console_level,
                'formatter': 'simple',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': file_level,
                'formatter': 'detailed',
                'filename': str(log_file_path),
                'maxBytes': 1000000,
                'backupCount': 5,
            },
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
        },
    }

    logging.config.dictConfig(LOGGING_CONFIG)
