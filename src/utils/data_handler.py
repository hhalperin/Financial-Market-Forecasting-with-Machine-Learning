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
        # Define the filename convention and full filepath
        filename = f"{ticker}_{data_type}_{date_range}.csv"
        filepath = os.path.join(self.data_dir, data_type, filename)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Cache key
        cache_key = filepath

        # Check cache first
        if cache_key in self.cache:
            logger.info(f"Loading data from cache for {filename}...")
            return self.cache[cache_key]

        # Load data from file if it exists
        if os.path.exists(filepath):
            logger.info(f"Loading existing data from {filepath}...")
            data = self.load_data(filepath, data_type)
        else:
            # Fetch new data if it doesn't exist
            logger.info(f"Fetching new data for {data_type}...")
            data = data_fetcher()
            if data is not None and not data.empty:
                self.save_data(data, filepath, data_type)

        # Store in cache
        self.cache[cache_key] = data
        return data

    def save_data(self, data, filepath, data_type):
        if data_type == 'embeddings':
            np.save(filepath, data)
        else:
            data.to_csv(filepath, index=False)
        logger.info(f"Data saved to: {filepath}")

    def load_data(self, filepath, data_type):
        if data_type == 'embeddings':
            return np.load(filepath, allow_pickle=True)
        else:
            return pd.read_csv(filepath)
