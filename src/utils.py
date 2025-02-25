# src/utils.py

import os
import yaml
import logging
import time
import psutil

def setup_logging(level=logging.INFO):
    """
    Configures a basic logging setup for the project.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    return logging.getLogger(__name__)

def load_config(config_path="config/base_config.yaml"):
    """
    Loads the YAML config file from the given path.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def measure_memory_and_time():
    """
    Returns current time, memory usage for performance tracking.
    """
    current_time = time.time()
    process = psutil.Process(os.getpid())
    current_mem = process.memory_info().rss / (1024 * 1024)
    return current_time, current_mem
