"""
Main Module for Data Processing

This script performs the following:
  - Loads configuration parameters from the centralized settings.
  - Uses the unified DataHandler instance (configured via config.py) to load and save data.
  - Processes price and news data using the DataProcessor.
  - Generates time horizon combinations via the TimeHorizonManager.
  - Saves the final preprocessed and numeric data for further use.
"""

from datetime import timedelta
import pandas as pd

from src.config import settings
from src.utils.logger import get_logger
from src.data_processing.data_processing import DataProcessor
from src.data_processing.time_horizon_manager import TimeHorizonManager
from src.utils.data_handler import DataHandler  # Import the DataHandler class

logger = get_logger("DataProcessingMain")

def main() -> None:
    """
    Main execution function that orchestrates the data processing pipeline.
    """
    # Log configuration details from settings
    logger.info(f"Configuration: ticker={settings.ticker}, start_date={settings.start_date}, "
                f"end_date={settings.end_date}, storage_mode={settings.storage_mode}")

    # Create an instance of DataHandler, so we can call data_handler.get_data(...)
    data_handler = DataHandler(
        bucket=settings.s3_bucket,
        base_data_dir=settings.data_storage_path,
        storage_mode=settings.storage_mode
    )

    ticker = settings.ticker
    date_range = f"{settings.start_date}_to_{settings.end_date}"
    logger.info(f"Processing data for ticker={ticker}, date_range={date_range}")

    # Dummy fetch function; replace with actual data fetching logic.
    def no_op_fetch() -> pd.DataFrame:
        return pd.DataFrame()

    # Now we can call data_handler.get_data(...) because we defined it in data_handler.py
    price_df = data_handler.get_data(ticker, date_range, "price", no_op_fetch, stage="price")
    news_df = data_handler.get_data(ticker, date_range, "news", no_op_fetch, stage="news")

    logger.info(f"Loaded price_df: shape={price_df.shape}, columns={list(price_df.columns)}")
    logger.info(f"Loaded news_df: shape={news_df.shape}, columns={list(news_df.columns)}")

    if price_df.empty or "DateTime" not in price_df.columns:
        logger.error("'DateTime' column missing or price_df is empty. Check aggregator output.")
        return
    if news_df.empty or "time_published" not in news_df.columns:
        logger.error("'time_published' column missing or news_df is empty. Check aggregator output.")
        return

    processor = DataProcessor(price_df, news_df)
    thm = TimeHorizonManager()
    combos = thm.generate_horizon_combos()
    logger.info(f"Generated {len(combos)} time horizon combinations.")

    # Determine maximum gather minutes from combos
    times = [int(c["gather_td"].total_seconds() // 60) for c in combos]
    max_gather_minutes = max(times) if times else 120

    # Generate a list of time horizons based on the maximum gather minutes
    time_horizons = [
        {"target_name": f"{m}_minutes", "time_horizon": timedelta(minutes=m)}
        for m in range(5, max_gather_minutes + 1, 5)
    ]
    processed_df = processor.process_pipeline(time_horizons)
    #logger.info(f"Final processed DataFrame (raw): shape={processed_df.shape}, columns={list(processed_df.columns)}")
    #logger.info(f"Numeric DataFrame for training: shape={processor.numeric_df.shape}, columns={list(processor.numeric_df.columns)}")

    # Save numeric and preprocessed data
    # We can store them by calling data_handler.get_data(...) or just directly saving:
    numeric_filename = data_handler.construct_filename(ticker, "numeric", date_range, "csv")
    data_handler.save_data(processor.numeric_df, numeric_filename, data_type="csv", stage="numeric")
    logger.info(f"Saved numeric data for training as {numeric_filename}.")

    # For embeddings (if you want them in .npy form):
    numeric_emb_filename = data_handler.construct_filename(ticker, "numeric", date_range, "npy")
    data_handler.save_data(processor.numeric_df.values, numeric_emb_filename, data_type="embeddings", stage="numeric")
    logger.info(f"Numeric embeddings saved to numeric/{numeric_emb_filename}.")

    # Similarly for preprocessed
    preprocessed_filename = data_handler.construct_filename(ticker, "preprocessed", date_range, "csv")
    data_handler.save_data(processed_df, preprocessed_filename, data_type="csv", stage="preprocessed")
    logger.info(f"Final preprocessed data saved to preprocessed/{preprocessed_filename}.")


if __name__ == "__main__":
    main()
