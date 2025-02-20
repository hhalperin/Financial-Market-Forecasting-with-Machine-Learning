"""
Main Module for Data Processing

This script performs the following:
  - Loads configuration parameters from the centralized settings.
  - Uses the unified DataHandler instance (configured via config.py) to load and save data.
  - Processes price and news data using the DataProcessor.
  - Generates time horizon combinations via the TimeHorizonManager.
  - Saves the final preprocessed and numeric data for further use.

Note: DataHandler is automatically configured (choosing local or S3) based on settings.
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
    # Log configuration details directly from settings.
    logger.info(f"Configuration: ticker={settings.ticker}, start_date={settings.start_date}, "
                f"end_date={settings.end_date}, storage_mode={settings.storage_mode}")

    # Instantiate DataHandler (instead of using the class directly).
    data_handler = DataHandler(
        bucket=settings.s3_bucket,
        base_data_dir=settings.data_storage_path,
        storage_mode=settings.storage_mode
    )

    # Construct a date range string and log it.
    ticker = settings.ticker
    date_range = f"{settings.start_date}_to_{settings.end_date}"
    logger.info(f"Processing data for ticker={ticker}, date_range={date_range}")

    # Dummy fetch function; replace with actual data fetching logic.
    def no_op_fetch() -> pd.DataFrame:
        return pd.DataFrame()

    # Retrieve price and news data using DataHandler's get_data method.
    price_df = data_handler(ticker, date_range, "price", no_op_fetch, stage="price")
    news_df = data_handler(ticker, date_range, "news", no_op_fetch, stage="news")

    # Log DataFrame information directly.
    logger.info(f"Loaded price_df: shape={price_df.shape}, columns={list(price_df.columns)}")
    logger.info(f"Loaded news_df: shape={news_df.shape}, columns={list(news_df.columns)}")

    # Check for critical columns.
    if price_df.empty or "DateTime" not in price_df.columns:
        logger.error("'DateTime' column missing or price_df is empty. Check aggregator output.")
        return
    if news_df.empty or "time_published" not in news_df.columns:
        logger.error("'time_published' column missing or news_df is empty. Check aggregator output.")
        return

    # Process data using DataProcessor.
    processor = DataProcessor(price_df, news_df)
    thm = TimeHorizonManager()
    combos = thm.generate_horizon_combos()
    logger.info(f"Generated {len(combos)} time horizon combinations.")

    # Determine maximum gather minutes from the combinations.
    times = [int(c["gather_td"].total_seconds() // 60) for c in combos]
    max_gather_minutes = max(times) if times else 120

    # Generate a list of time horizons based on the maximum gather minutes.
    time_horizons = [
        {"target_name": f"{m}_minutes", "time_horizon": timedelta(minutes=m)}
        for m in range(5, max_gather_minutes + 1, 5)
    ]
    processed_df = processor.process_pipeline(time_horizons)
    #logger.info(f"Final processed DataFrame (raw): shape={processed_df.shape}, columns={list(processed_df.columns)}")
    #logger.info(f"Numeric DataFrame for training: shape={processor.numeric_df.shape}, columns={list(processor.numeric_df.columns)}")

    # Save numeric and preprocessed data.
    data_handler(ticker, date_range, "numeric", lambda: processor.numeric_df, stage="numeric")
    #logger.info("Saved numeric data for training.")

    numeric_filename = data_handler.construct_filename(ticker, "numeric", date_range, "npy")
    data_handler.save_data(processor.numeric_df.values, numeric_filename, data_type="embeddings", stage="numeric")
    #logger.info(f"Numeric embeddings saved to numeric/{numeric_filename}.")

    data_handler(ticker, date_range, "preprocessed", lambda: processed_df, stage="preprocessed")
    #logger.info("Saved final preprocessed data.")


if __name__ == "__main__":
    main()
