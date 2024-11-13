import os
import pandas as pd
import numpy as np
from logger import get_logger

logger = get_logger('DataHandler')

class DataHandler:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.cache = {}  # In-memory cache to speed up repeated loads

    def __call__(self, ticker, date_range, data_type, data_fetcher, config_id="default"):
        filename = f"{ticker}_{date_range}_{data_type}_{config_id}"
        filepath = os.path.join(self.data_dir, data_type, f"{filename}")

        # Create directories if not exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Cache key
        cache_key = filepath

        # Check cache first
        if cache_key in self.cache:
            logger.info(f"Loading data from cache for {filename}...")
            return self.cache[cache_key]

        if data_type == 'embeddings':
            filepath += ".npy"
        else:
            filepath += ".csv"

        if os.path.exists(filepath):
            logger.info(f"Loading existing data from {filepath}...")
            data = self.load_data(filepath, data_type)
        else:
            logger.info(f"Fetching new data for {data_type}...")
            data = data_fetcher()
            self.save_data(data, filepath, data_type)

        # Store in cache
        self.cache[cache_key] = data
        return data
