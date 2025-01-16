import os
import io
import json
import pandas as pd
import numpy as np
import boto3
from utils.logger import get_logger

logger = get_logger('DataHandler')

class DataHandler:
    def __init__(self, bucket=None, base_data_dir='../data', storage_mode='local'):
        """
        :param bucket: S3 bucket name (if storage_mode='s3').
        :param base_data_dir: The top-level folder for local data storage (default='data').
        :param storage_mode: 's3' or 'local'.
        """
        self.bucket = bucket
        self.base_data_dir = base_data_dir
        self.storage_mode = storage_mode.lower()

        if self.storage_mode == 's3':
            self.s3 = boto3.client('s3')
        else:
            os.makedirs(self.base_data_dir, exist_ok=True)

        self.cache = {}

    def __call__(self, ticker, date_range, data_type, data_fetcher, stage='raw'):
        filename_ext = '.npy' if data_type == 'embeddings' else '.csv'
        filename = f"{ticker}_{data_type}_{date_range}{filename_ext}"
        return self._load_or_fetch(filename, data_type, data_fetcher, stage)

    def _load_or_fetch(self, filename, data_type, data_fetcher, stage):
        cache_key = f"{stage}/{filename}"

        if cache_key in self.cache:
            logger.info(f"Loading from cache: {cache_key}")
            return self.cache[cache_key]

        data = self.load_data(filename, data_type, stage)
        if data is not None:
            self.cache[cache_key] = data
            return data

        data = data_fetcher()
        self._validate_data(data, data_type)
        self.save_data(data, filename, data_type, stage)
        self.cache[cache_key] = data
        return data

    def save_data(self, data, filename, data_type, stage):
        if self.storage_mode == 's3':
            self._save_s3(data, filename, data_type, stage)
        else:
            self._save_local(data, filename, data_type, stage)

    def load_data(self, filename, data_type, stage):
        if self.storage_mode == 's3':
            return self._load_s3(filename, data_type, stage)
        else:
            return self._load_local(filename, data_type, stage)

    def _validate_data(self, data, data_type):
        if data is None:
            raise ValueError("Data fetcher returned None.")
        if data_type == 'embeddings' and not isinstance(data, np.ndarray):
            raise ValueError("Expected NumPy array for embeddings.")
        if data_type != 'embeddings' and not isinstance(data, pd.DataFrame):
            raise ValueError("Expected Pandas DataFrame for non-embedding data.")

    def _save_s3(self, data, filename, data_type, stage):
        s3_key = f"{stage}/{filename}"
        buffer = io.BytesIO() if data_type == 'embeddings' else io.StringIO()
        if data_type == 'embeddings':
            np.save(buffer, data)
            buffer.seek(0)
        else:
            data.to_csv(buffer, index=False)
        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=buffer.getvalue())
        logger.info(f"Data saved to S3: {s3_key}")

    def _load_s3(self, filename, data_type, stage):
        s3_key = f"{stage}/{filename}"
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            if data_type == 'embeddings':
                buffer = io.BytesIO(obj['Body'].read())
                return np.load(buffer, allow_pickle=True)
            else:
                return pd.read_csv(obj['Body'])
        except self.s3.exceptions.NoSuchKey:
            logger.info(f"No existing S3 data for {s3_key}")
            return None

    def _save_local(self, data, filename, data_type, stage):
        local_path = self._prepare_local_path(filename, stage)
        if data_type == 'embeddings':
            np.save(local_path, data)
        else:
            data.to_csv(local_path, index=False)
        logger.info(f"Data saved locally: {local_path}")

    def _load_local(self, filename, data_type, stage):
        local_path = self._prepare_local_path(filename, stage, create=False)
        if not os.path.exists(local_path):
            logger.info(f"No existing local file: {local_path}")
            return None
        if data_type == 'embeddings':
            return np.load(local_path, allow_pickle=True)
        else:
            return pd.read_csv(local_path)

    def _prepare_local_path(self, filename, stage, create=True):
        stage_dir = os.path.join(self.base_data_dir, stage)
        if create:
            try:
                os.makedirs(stage_dir, exist_ok=True)
                logger.debug(f"Created directory: {stage_dir}")
            except Exception as e:
                logger.error(f"Failed to create directory {stage_dir}: {e}")
        return os.path.join(stage_dir, filename)


    def save_bytes(self, data_bytes, filename, stage='models'):
        path = self._prepare_local_path(filename, stage)
        if self.storage_mode == 's3':
            self.s3.put_object(Bucket=self.bucket, Key=f"{stage}/{filename}", Body=data_bytes)
            logger.info(f"File saved to S3: {filename}")
        else:
            with open(path, 'wb') as f:
                f.write(data_bytes)
            logger.info(f"File saved locally: {path}")
