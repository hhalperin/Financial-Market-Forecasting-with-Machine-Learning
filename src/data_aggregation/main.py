# src/data_aggregation/main.py

import os
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.utils.data_handler import DataHandler
from src.data_aggregation.data_aggregator import DataAggregator
import pandas as pd

logger = get_logger("DataAggregationMain")

def load_config():
    """
    Load environment variables from .env.
    """
    load_dotenv()
    local_mode = os.getenv("LOCAL_MODE", "false").lower() == "true"
    storage_mode = "local" if local_mode else "s3"
    s3_bucket = os.getenv("S3_BUCKET") if not local_mode else None

    if not local_mode and not s3_bucket:
        raise ValueError("S3_BUCKET must be set when LOCAL_MODE=false.")

    ticker = os.getenv("TICKER", "NVDA")
    start_date = os.getenv("START_DATE", "2022-01-01")
    end_date = os.getenv("END_DATE", "2024-01-31")
    interval = os.getenv("INTERVAL", "1min")

    return {
        "local_mode": local_mode,
        "storage_mode": storage_mode,
        "s3_bucket": s3_bucket,
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval
    }

def setup_data_handler(config):
    if config["local_mode"]:
        logger.info("[SETUP] Using local file storage.")
        return DataHandler(storage_mode="local")
    else:
        logger.info(f"[SETUP] Using S3 file storage with bucket: {config['s3_bucket']}")
        return DataHandler(bucket=config["s3_bucket"], storage_mode="s3")

def setup_aggregator(config, data_handler):
    return DataAggregator(
        ticker=config["ticker"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        interval=config["interval"],
        data_handler=data_handler,
        local_mode=config["local_mode"]
    )

def main():
    logger.info("[MAIN] Loading configuration...")
    config = load_config()
    logger.info(f"[MAIN] Configuration loaded: {config}")

    # Build DataHandler
    data_handler = setup_data_handler(config)

    # Build aggregator and fetch data
    aggregator = setup_aggregator(config, data_handler)
    logger.info("[MAIN] Starting data aggregation (monthly intraday + news)...")
    price_df, news_df = aggregator.aggregate_data()

    logger.info("[MAIN] Data aggregation process completed.")

    # Save the aggregated data using DataHandler
    ticker = config["ticker"]
    start_date = config["start_date"]
    end_date = config["end_date"]

    # Price data
    if price_df is not None and not price_df.empty:
        price_filename = f"{ticker}_price_{start_date}_to_{end_date}.csv"
        data_handler.save_data(price_df, price_filename, data_type="csv", stage="price")
        logger.info(f"Saved price data to {price_filename}")
    else:
        logger.warning("price_df is empty. Skipping price CSV save.")

    # News data
    if news_df is not None and not news_df.empty:
        news_filename = f"{ticker}_news_{start_date}_to_{end_date}.csv"
        data_handler.save_data(news_df, news_filename, data_type="csv", stage="news")
        logger.info(f"Saved news data to {news_filename}")
    else:
        logger.warning("news_df is empty. Skipping news CSV save.")

if __name__ == "__main__":
    main()
