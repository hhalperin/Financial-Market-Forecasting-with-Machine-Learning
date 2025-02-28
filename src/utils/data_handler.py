"""
DataHandler Module

This module provides a DataHandler class for managing data storage and retrieval.
It supports both local file storage and S3 storage. Data are saved as CSV files (or npy for embeddings).
For very large datasets, consider switching to Parquet for improved I/O performance.
This module also caches loaded data to reduce redundant I/O operations.
"""

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

logger = get_logger("DataHandler")


class DataHandler:
    def __init__(self, bucket: Union[str, None] = None, base_data_dir: Union[str, None] = None, storage_mode: str = "local") -> None:
        """
        Initializes the DataHandler.

        :param bucket: S3 bucket name (if using S3).
        :param base_data_dir: Base directory for local storage; defaults to settings.data_storage_path.
        :param storage_mode: Mode of storage ("local" or "s3").
        """
        if base_data_dir is None:
            base_data_dir = settings.data_storage_path
        self.bucket = bucket
        self.base_data_dir = os.path.normpath(base_data_dir)
        self.storage_mode = storage_mode.lower()
        if self.storage_mode == "s3":
            self.s3 = boto3.client("s3")
        else:
            os.makedirs(self.base_data_dir, exist_ok=True)
        # Cache to reduce redundant I/O.
        self.cache: dict[str, Any] = {}
        logger.info(f"Initialized DataHandler with base_data_dir={self.base_data_dir}")

    def construct_filename(self, ticker: str, stage: str, date_range: str, file_type: str) -> str:
        """
        Constructs a standardized filename.

        :param ticker: Stock ticker symbol.
        :param stage: Data stage (e.g., "raw", "numeric").
        :param date_range: Date range string.
        :param file_type: File extension (e.g., "csv", "npy").
        :return: Constructed filename.
        """
        return f"{ticker}_{stage}_{date_range}.{file_type}"

    def __call__(self, ticker: str, date_range: str, data_type: str, data_fetcher, stage: str = "raw") -> Any:
        """
        Main fetch-or-load logic.

        :param ticker: Stock ticker symbol.
        :param date_range: Date range string.
        :param data_type: Data type ("embeddings" for npy, else assumed csv).
        :param data_fetcher: Callable to fetch data if not present.
        :param stage: Data stage.
        :return: Loaded or fetched data.
        """
        file_type = "npy" if data_type == "embeddings" else "csv"
        filename = self.construct_filename(ticker, stage, date_range, file_type)
        return self._load_or_fetch(filename, data_type, data_fetcher, stage)

    def get_data(self, ticker: str, date_range: str, data_type: str, data_fetcher, stage: str = "raw") -> Any:
        """
        Alternative method to retrieve data; delegates to __call__.

        :param ticker: Stock ticker symbol.
        :param date_range: Date range string.
        :param data_type: Data type.
        :param data_fetcher: Callable to fetch data.
        :param stage: Data stage.
        :return: Data loaded from storage or freshly fetched.
        """
        return self.__call__(ticker, date_range, data_type, data_fetcher, stage)

    def _load_or_fetch(self, filename: str, data_type: str, data_fetcher, stage: str) -> Any:
        """
        Attempts to load data from storage; if not available, fetches and saves it.

        :param filename: Filename.
        :param data_type: Data type.
        :param data_fetcher: Callable to fetch data.
        :param stage: Data stage.
        :return: Data.
        """
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
        Saves data using the chosen storage backend.

        :param data: Data to save.
        :param filename: Filename.
        :param data_type: Data type.
        :param stage: Data stage.
        """
        if self.storage_mode == "s3":
            self._save_s3(data, filename, data_type, stage)
        else:
            self._save_local(data, filename, data_type, stage)

    def load_data(self, filename: str, data_type: str, stage: str) -> Any:
        """
        Loads data from storage.

        :param filename: Filename.
        :param data_type: Data type.
        :param stage: Data stage.
        :return: Loaded data.
        """
        if self.storage_mode == "s3":
            return self._load_s3(filename, data_type, stage)
        else:
            return self._load_local(filename, data_type, stage)

    def _validate_data(self, data: Any, data_type: str) -> None:
        """
        Validates that data is not None and of the expected type.

        :param data: Data to validate.
        :param data_type: Expected data type.
        :raises ValueError: If validation fails.
        """
        if data is None:
            raise ValueError("Data fetcher returned None.")
        if data_type == "embeddings" and not isinstance(data, np.ndarray):
            raise ValueError("Expected NumPy array for embeddings.")
        if data_type != "embeddings" and not isinstance(data, pd.DataFrame):
            raise ValueError("Expected Pandas DataFrame for non-embedding data.")

    def _save_s3(self, data: Union[pd.DataFrame, np.ndarray], filename: str, data_type: str, stage: str) -> None:
        """
        Saves data to S3.

        :param data: Data to save.
        :param filename: Filename.
        :param data_type: Data type.
        :param stage: Data stage.
        """
        s3_key = f"{stage}/{filename}"
        if data_type == "embeddings":
            buffer = io.BytesIO()
            np.save(buffer, data)
            buffer.seek(0)
        else:
            buffer = io.StringIO()
            data.to_csv(buffer, index=False)
        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=buffer.getvalue())

    def _load_s3(self, filename: str, data_type: str, stage: str) -> Any:
        """
        Loads data from S3.

        :param filename: Filename.
        :param data_type: Data type.
        :param stage: Data stage.
        :return: Loaded data or None if not found.
        """
        s3_key = f"{stage}/{filename}"
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            if data_type == "embeddings":
                buffer = io.BytesIO(obj["Body"].read())
                return np.load(buffer, allow_pickle=True)
            else:
                return pd.read_csv(obj["Body"])
        except self.s3.exceptions.NoSuchKey:
            logger.info(f"No existing S3 data for {s3_key}")
            return None

    def _save_local(self, data: Union[pd.DataFrame, np.ndarray], filename: str, data_type: str, stage: str) -> None:
        """
        Saves data to a local file.

        :param data: Data to save.
        :param filename: Filename.
        :param data_type: Data type.
        :param stage: Data stage.
        """
        local_path = self._prepare_local_path(filename, stage)
        if data_type == "embeddings":
            np.save(local_path, data)
        else:
            data.to_csv(local_path, index=False)

    def _load_local(self, filename: str, data_type: str, stage: str) -> Any:
        """
        Loads data from a local file.

        :param filename: Filename.
        :param data_type: Data type.
        :param stage: Data stage.
        :return: Loaded data or an empty DataFrame if file is missing or empty.
        """
        local_path = self._prepare_local_path(filename, stage, create=False)
        if not os.path.exists(local_path):
            logger.info(f"No existing local file: {local_path}")
            return None
        if os.path.getsize(local_path) == 0:
            logger.warning(f"Local file exists but is empty: {local_path}")
            return pd.DataFrame()
        if data_type == "embeddings":
            return np.load(local_path, allow_pickle=True)
        else:
            try:
                return pd.read_csv(local_path)
            except pd.errors.EmptyDataError:
                logger.warning(f"EmptyDataError reading file {local_path}")
                return pd.DataFrame()

    def _prepare_local_path(self, filename: str, stage: str, create: bool = True) -> str:
        """
        Prepares the full path for a given stage and filename.

        :param filename: Filename.
        :param stage: Data stage.
        :param create: If True, creates the directory if it does not exist.
        :return: Full file path.
        """
        stage_dir = os.path.join(self.base_data_dir, stage)
        if create:
            os.makedirs(stage_dir, exist_ok=True)
        return os.path.join(stage_dir, filename)

    def save_dataframe(self, df: pd.DataFrame, filename: str, stage: str) -> None:
        """
        Saves a DataFrame using the appropriate storage backend.

        :param df: DataFrame to save.
        :param filename: Filename.
        :param stage: Data stage.
        """
        if self.storage_mode == "s3":
            self._save_s3(df, filename, data_type="csv", stage=stage)
        else:
            self._save_local(df, filename, data_type="csv", stage=stage)

    def save_figure_bytes(self, figure_bytes: bytes, filename: str, stage: str) -> None:
        """
        Saves figure bytes (e.g., from a matplotlib plot) using the appropriate storage backend.

        :param figure_bytes: Figure data in bytes.
        :param filename: Filename.
        :param stage: Data stage.
        """
        if self.storage_mode == "s3":
            s3_key = f"{stage}/{filename}"
            self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=figure_bytes)
        else:
            stage_dir = os.path.join(self.base_data_dir, stage)
            os.makedirs(stage_dir, exist_ok=True)
            local_path = os.path.join(stage_dir, filename)
            with open(local_path, "wb") as f:
                f.write(figure_bytes)

    def save_model(self, model: torch.nn.Module, filepath: str) -> None:
        """
        Saves a PyTorch model's state dictionary to a file.

        :param model: PyTorch model.
        :param filepath: File path to save the model.
        """
        if self.storage_mode == "s3":
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            s3_key = os.path.relpath(filepath, self.base_data_dir).replace("\\", "/")
            self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=buffer.read())
        else:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(model.state_dict(), filepath)

    def save_json(self, data: Any, filename: str, stage: str) -> None:
        """
        Saves a Python object as a JSON file.

        :param data: Data to save.
        :param filename: Filename.
        :param stage: Data stage.
        """
        if self.storage_mode == "s3":
            json_str = json.dumps(data, indent=4)
            s3_key = f"{stage}/{filename}"
            self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=json_str)
        else:
            stage_dir = os.path.join(self.base_data_dir, stage)
            os.makedirs(stage_dir, exist_ok=True)
            local_path = os.path.join(stage_dir, filename)
            with open(local_path, "w") as f:
                json.dump(data, f, indent=4)
