import json
import numpy as np
from datetime import timedelta
from data_processing import PreprocessingManager, TimeHorizonManager, DataEmbedder
from utils.logger import get_logger
from sklearn.model_selection import train_test_split
from utils.data_handler import DataHandler

class ModelPipeline:
    def __init__(self, ticker, start_date, end_date, use_batch_api=False, s3_bucket=None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.use_batch_api = use_batch_api
        self.logger = get_logger('ModelPipeline')
        self.data_handler = DataHandler(s3_bucket)
        self.s3_prefix = f"{ticker}_{start_date}_to_{end_date}/"

    def handle_embeddings(self, preprocessed_df, config_id):
        embeddings_key = f"{self.s3_prefix}{config_id}_embeddings.npy"

        # If using batch APIs, skip for brevity (assume local embeddings)
        if not self.use_batch_api:
            self.logger.info("Generating embeddings synchronously...")
            embedder = DataEmbedder(
                model_type='nvidia', 
                model_name='sentence-transformers/all-MiniLM-L6-v2', 
                n_components=128, 
                use_batch_api=False
            )
            embeddings = embedder.create_embeddings_from_dataframe(
                preprocessed_df, 
                self.ticker, 
                f"{self.start_date}_to_{self.end_date}", 
                self.data_handler, 
                config_id=config_id
            )
            # embeddings already saved by data_handler in create_embeddings_from_dataframe method
            return embeddings
        else:
            # Handle batch scenario (not shown fully for brevity)
            # Possibly retrieve embeddings from a previous batch job result on S3.
            embeddings = self.data_handler.load_data(embeddings_key, data_type='embeddings')
            return embeddings

    def train_and_evaluate_models(self, X, preprocessed_df, time_horizons, model_manager):
        # Example of using already processed X, no local IO
        for config in time_horizons:
            target_name = config['target_name']
            time_horizon = config['time_horizon']

            self.logger.info(f"Training model for horizon: {time_horizon}")
            pm = PreprocessingManager(preprocessed_df)
            pm.df = pm.df.dropna(subset=['Close'])  # ensure no NaNs
            pm.calculate_dynamic_targets(column_name='Close', target_configs=[config])

            if target_name not in pm.df.columns:
                self.logger.error(f"Target '{target_name}' not found.")
                continue

            y = pm.df[target_name].fillna(0).values

            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            # Train and evaluate model - no local file usage, model artifacts in S3
            model_manager.train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, timestamps=range(len(y_test)))
