from __future__ import annotations

from pathlib import Path

import dotenv
import yaml


def setup_project_env(
        config_filename='my_config.yaml', env_filename='.env'):
    """
    Set up the project environment and load configuration.

    config_filename: Name of the configuration file.
    env_filename: Name of the .env file.
    """
    # Set up the environment
    project_dir = Path(__file__).resolve().parents[1]
    dotenv_path = project_dir / env_filename
    dotenv.load_dotenv(dotenv_path)

    # Load configuration
    config_path = project_dir / config_filename
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    return project_dir, config


if __name__ == '__main__':
    project_dir, config = setup_project_env()
