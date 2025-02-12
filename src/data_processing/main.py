"""
Main Module for Data Processing

This script loads configuration and data via a DataHandler, processes the data using the DataProcessor,
and saves the resulting preprocessed and numeric DataFrames.
"""

import os
from datetime import timedelta
from typing import Dict
import pandas as pd
from src.config import Settings
from src.utils.logger import get_logger
from src.utils.data_handler import DataHandler
from src.data_processing.data_processing import DataProcessor
from src.data_processing.time_horizon_manager import TimeHorizonManager

logger = get_logger("DataProcessingMain")

def load_config() -> Dict[str, str]:
    """
    Loads configuration from Settings.
    """
    settings = Settings()
    s3_bucket = settings.s3_bucket.strip()
    local_mode = settings.local_mode if s3_bucket == "" else False
    storage_mode = "local" if local_mode else "s3"
    return {
        "local_mode": local_mode,
        "storage_mode": storage_mode,
        "ticker": settings.ticker,
        "start_date": settings.start_date,
        "end_date": settings.end_date,
        "interval": settings.interval,
        "s3_bucket": s3_bucket
    }

def build_data_handler(config: Dict[str, str]) -> DataHandler:
    """
    Initializes the DataHandler based on the storage mode.
    """
    if config["storage_mode"] == "s3":
        return DataHandler(bucket=config["s3_bucket"], storage_mode="s3")
    else:
        return DataHandler(storage_mode="local")

def print_dataframe_info(label: str, df: pd.DataFrame) -> None:
    """
    Logs basic information about a DataFrame.
    """
    logger.info(f"[{label}] shape={df.shape}, columns={list(df.columns)}")

def main() -> None:
    """
    Main function for the data processing pipeline.
    """
    config = load_config()
    logger.info(f"[DataProcessingMain] config: {config}")
    data_handler = build_data_handler(config)

    ticker = config["ticker"]
    date_range = f"{config['start_date']}_to_{config['end_date']}"
    logger.info(f"Processing for ticker={ticker}, date_range={date_range}")

    # Define a no-op fetch function to be used by the DataHandler (assumes data is already stored).
    def no_op_fetch() -> pd.DataFrame:
        return pd.DataFrame()

    # Load price and news data using DataHandler.
    price_df = data_handler(ticker, date_range, "price", no_op_fetch, stage='price')
    news_df = data_handler(ticker, date_range, "news", no_op_fetch, stage='news')

    print_dataframe_info("Loaded price_df", price_df)
    print_dataframe_info("Loaded news_df", news_df)

    # Validate that essential columns exist.
    if price_df.empty or "DateTime" not in price_df.columns:
        logger.error("'DateTime' column missing or price_df is empty. Check aggregator output.")
        return
    if news_df.empty or "time_published" not in news_df.columns:
        logger.error("'time_published' column missing or news_df is empty. Check aggregator output.")
        return

    # Initialize the DataProcessor and process the pipeline.
    processor = DataProcessor(price_df, news_df)
    thm = TimeHorizonManager()
    combos = thm.generate_horizon_combos()
    logger.info(f"Generated {len(combos)} horizon combos, e.g. {combos[:5]}...")

    times = [int(c["gather_td"].total_seconds() // 60) for c in combos]
    max_gather_minutes = max(times) if times else 120

    time_horizons = [
        {"target_name": f"{m}_minutes", "time_horizon": timedelta(minutes=m)} 
        for m in range(5, max_gather_minutes + 1, 5)
    ]
    processed_df: pd.DataFrame = processor.process_pipeline(time_horizons)
    print_dataframe_info("Final processed DataFrame (raw)", processed_df)
    print_dataframe_info("Numeric DataFrame for training", processor.numeric_df)

    # Save numeric DataFrame (with embeddings) as CSV.
    def fetch_numeric() -> pd.DataFrame:
        return processor.numeric_df
    _ = data_handler(ticker, date_range, "numeric", fetch_numeric, stage='numeric')
    logger.info("[DataProcessingMain] Saved numeric data for training.")

    # Optionally, save numeric data as a NumPy file.
    numeric_filename = data_handler.construct_filename(ticker, "numeric", date_range, "npy")
    data_handler.save_data(processor.numeric_df.values, numeric_filename, data_type="embeddings", stage='numeric')
    logger.info(f"Numeric embeddings saved to numeric/{numeric_filename}.")

    # Save final preprocessed data.
    def fetch_preprocessed() -> pd.DataFrame:
        return processed_df
    _ = data_handler(ticker, date_range, "preprocessed", fetch_preprocessed, stage='preprocessed')
    logger.info("[DataProcessingMain] Saved final preprocessed data.")

if __name__ == "__main__":
    main()
