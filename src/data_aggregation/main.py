"""
Main Module for Data Aggregation

Loads configuration, sets up the data handler and aggregator, executes the data aggregation process,
and saves the resulting data to the appropriate storage (local or S3).
"""

import os
from dotenv import load_dotenv
import pandas as pd
from src.utils.logger import get_logger
from src.utils.data_handler import DataHandler
from src.data_aggregation.data_aggregator import DataAggregator
from src.config import Settings  # Centralized configuration

logger = get_logger("DataAggregationMain")


def load_config() -> dict:
    """
    Loads configuration settings from the Settings model.

    :return: A dictionary containing configuration parameters.
    """
    settings = Settings()
    config = {
        "local_mode": settings.local_mode,
        "storage_mode": "local" if settings.local_mode else "s3",
        "s3_bucket": settings.s3_bucket if not settings.local_mode else None,
        "ticker": settings.ticker,
        "start_date": settings.start_date,
        "end_date": settings.end_date,
        "interval": settings.interval
    }
    return config


def setup_data_handler(config: dict) -> DataHandler:
    """
    Initializes the data handler based on the storage mode.

    :param config: Configuration dictionary.
    :return: An instance of DataHandler configured for local or S3 storage.
    """
    if config["local_mode"]:
        logger.info("[SETUP] Using local file storage.")
        return DataHandler(storage_mode="local")
    else:
        logger.info(f"[SETUP] Using S3 file storage with bucket: {config['s3_bucket']}")
        return DataHandler(bucket=config["s3_bucket"], storage_mode="s3")


def setup_aggregator(config: dict, data_handler: DataHandler) -> DataAggregator:
    """
    Initializes the DataAggregator with the provided configuration and data handler.

    :param config: Configuration dictionary.
    :param data_handler: DataHandler instance.
    :return: An instance of DataAggregator.
    """
    return DataAggregator(
        ticker=config["ticker"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        interval=config["interval"],
        data_handler=data_handler,
        local_mode=config["local_mode"]
    )


def main() -> None:
    """
    Main function that orchestrates the data aggregation process:
    1. Loads configuration.
    2. Sets up the data handler.
    3. Aggregates data.
    4. Saves the data to storage.
    """
    logger.info("[MAIN] Loading configuration...")
    config = load_config()
    logger.info(f"[MAIN] Configuration loaded: {config}")

    data_handler = setup_data_handler(config)
    aggregator = setup_aggregator(config, data_handler)

    logger.info("[MAIN] Starting data aggregation (monthly intraday + news)...")
    price_df, news_df = aggregator.aggregate_data()
    logger.info("[MAIN] Data aggregation process completed.")

    ticker = config["ticker"]
    start_date = config["start_date"]
    end_date = config["end_date"]

    # Save price data if available.
    if price_df is not None and not price_df.empty:
        price_filename = f"{ticker}_price_{start_date}_to_{end_date}.csv"
        data_handler.save_data(price_df, price_filename, data_type="csv", stage="price")
        logger.info(f"[MAIN] Saved price data to {price_filename}")
    else:
        logger.warning("[MAIN] Price DataFrame is empty. Skipping CSV save for price data.")

    # Save news data if available.
    if news_df is not None and not news_df.empty:
        news_filename = f"{ticker}_news_{start_date}_to_{end_date}.csv"
        data_handler.save_data(news_df, news_filename, data_type="csv", stage="news")
        logger.info(f"[MAIN] Saved news data to {news_filename}")
    else:
        logger.warning("[MAIN] News DataFrame is empty. Skipping CSV save for news data.")


if __name__ == "__main__":
    main()
