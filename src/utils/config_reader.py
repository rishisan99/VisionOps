# src/utils/config_reader.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.exception.exception import CustomException
from src.logging.logger import logging


def read_yaml_config(config_path: Path) -> Dict[str, Any]:
    try:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError("Config YAML must load into a dict")

        return data

    except Exception as e:
        logging.error(f"Failed to read config: {e}")
        raise CustomException(e)
