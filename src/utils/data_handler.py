import os
import io
import json
import pandas as pd
import numpy as np
import boto3
import torch
from typing import Any, Union
from src.utils.logger import get_logger
from src.config import settings  # Import global settings

logger = get_logger('DataHandler')

class DataHandler:
    def __init__(self, bucket: Union[str, None] = None, base_data_dir: Union[str, None] = None, storage_mode: str = 'local') -> None:
        # Use the global settings for the base data directory if not provided.
        if base_data_dir is None:
            base_data_dir = settings.data_storage_path
        self.bucket = bucket
        self.base_data_dir = os.path.normpath(base_data_dir)
        self.storage_mode = storage_mode.lower()
        if self.storage_mode == 's3':
            self.s3 = boto3.client('s3')
        else:
            os.makedirs(self.base_data_dir, exist_ok=True)
        self.cache: dict[str, Any] = {}
        logger.info(f"Initialized DataHandler with base_data_dir={self.base_data_dir}")

    def construct_filename(self, ticker: str, stage: str, date_range: str, file_type: str) -> str:
        """
        Constructs a deterministic filename.
        """
        return f"{ticker}_{stage}_{date_range}.{file_type}"

    def __call__(self, ticker: str, date_range: str, data_type: str, data_fetcher, stage: str = 'raw') -> Any:
        file_type = 'npy' if data_type == 'embeddings' else 'csv'
        filename = self.construct_filename(ticker, stage, date_range, file_type)
        return self._load_or_fetch(filename, data_type, data_fetcher, stage)

    def _load_or_fetch(self, filename: str, data_type: str, data_fetcher, stage: str) -> Any:
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

    def save_data(self, data: Union[pd.DataFrame, np.ndarray], filename: str, data_type: str, stage: str) -> None:
        """
        Saves data locally or to S3.
        """
        if self.storage_mode == 's3':
            self._save_s3(data, filename, data_type, stage)
        else:
            self._save_local(data, filename, data_type, stage)

    def load_data(self, filename: str, data_type: str, stage: str) -> Any:
        """
        Loads data locally or from S3.
        """
        if self.storage_mode == 's3':
            return self._load_s3(filename, data_type, stage)
        else:
            return self._load_local(filename, data_type, stage)

    def _validate_data(self, data: Any, data_type: str) -> None:
        if data is None:
            raise ValueError("Data fetcher returned None.")
        if data_type == 'embeddings' and not isinstance(data, np.ndarray):
            raise ValueError("Expected NumPy array for embeddings.")
        if data_type != 'embeddings' and not isinstance(data, pd.DataFrame):
            raise ValueError("Expected Pandas DataFrame for non-embedding data.")

    def _save_s3(self, data: Union[pd.DataFrame, np.ndarray], filename: str, data_type: str, stage: str) -> None:
        s3_key = f"{stage}/{filename}"
        if data_type == 'embeddings':
            buffer = io.BytesIO()
            np.save(buffer, data)
            buffer.seek(0)
        else:
            buffer = io.StringIO()
            data.to_csv(buffer, index=False)
        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=buffer.getvalue())
        logger.info(f"Data saved to S3: {s3_key}")

    def _load_s3(self, filename: str, data_type: str, stage: str) -> Any:
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

    def _save_local(self, data: Union[pd.DataFrame, np.ndarray], filename: str, data_type: str, stage: str) -> None:
        local_path = self._prepare_local_path(filename, stage)
        if data_type == 'embeddings':
            np.save(local_path, data)
        else:
            data.to_csv(local_path, index=False)
        logger.info(f"Data saved locally: {local_path}")

    def _load_local(self, filename: str, data_type: str, stage: str) -> Any:
        local_path = self._prepare_local_path(filename, stage, create=False)
        if not os.path.exists(local_path):
            logger.info(f"No existing local file: {local_path}")
            return None
        if data_type == 'embeddings':
            return np.load(local_path, allow_pickle=True)
        else:
            return pd.read_csv(local_path)

    def _prepare_local_path(self, filename: str, stage: str, create: bool = True) -> str:
        stage_dir = os.path.join(self.base_data_dir, stage)
        if create:
            os.makedirs(stage_dir, exist_ok=True)
        return os.path.join(stage_dir, filename)
    
    def save_dataframe(self, df: pd.DataFrame, filename: str, stage: str) -> None:
        if self.storage_mode == "s3":
            self._save_s3(df, filename, data_type="csv", stage=stage)
        else:
            self._save_local(df, filename, data_type="csv", stage=stage)

    def save_figure_bytes(self, figure_bytes: bytes, filename: str, stage: str) -> None:
        if self.storage_mode == "s3":
            s3_key = f"{stage}/{filename}"
            self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=figure_bytes)
            logger.info(f"Figure saved to S3: {s3_key}")
        else:
            stage_dir = os.path.join(self.base_data_dir, stage)
            os.makedirs(stage_dir, exist_ok=True)
            local_path = os.path.join(stage_dir, filename)
            with open(local_path, "wb") as f:
                f.write(figure_bytes)
            # logger.info(f"Figure saved locally: {local_path}")

    def save_model(self, model: torch.nn.Module, filepath: str) -> None:
        if self.storage_mode == "s3":
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            s3_key = os.path.relpath(filepath, self.base_data_dir).replace("\\", "/")
            self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=buffer.read())
            logger.info(f"Model saved to S3: {s3_key}")
        else:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(model.state_dict(), filepath)
            logger.info(f"Model saved locally: {filepath}")

    def save_json(self, data: Any, filename: str, stage: str) -> None:
        if self.storage_mode == "s3":
            json_str = json.dumps(data, indent=4)
            s3_key = f"{stage}/{filename}"
            self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=json_str)
            logger.info(f"JSON data saved to S3: {s3_key}")
        else:
            stage_dir = os.path.join(self.base_data_dir, stage)
            os.makedirs(stage_dir, exist_ok=True)
            local_path = os.path.join(stage_dir, filename)
            with open(local_path, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"JSON data saved locally: {local_path}")