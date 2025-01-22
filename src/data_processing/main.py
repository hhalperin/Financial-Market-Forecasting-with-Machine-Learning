# src/data_processing_main.py

import os
from dotenv import load_dotenv
import pandas as pd
from src.utils.logger import get_logger
from src.utils.data_handler import DataHandler
from .data_processing import DataProcessor
from .time_horizon_manager import TimeHorizonManager
from datetime import timedelta

logger = get_logger("DataProcessingMain")

def load_config():
    """
    Load environment variables from .env.
    """
    load_dotenv()
    local_mode = os.getenv("LOCAL_MODE", "false").lower() == "true"
    storage_mode = "local" if local_mode else "s3"

    ticker = os.getenv("TICKER", "NVDA")
    start_date = os.getenv("START_DATE", "2022-01-01")
    end_date = os.getenv("END_DATE", "2024-01-31")
    return {
        "local_mode": local_mode,
        "storage_mode": storage_mode,
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "s3_bucket": os.getenv("S3_BUCKET", "")  # only used if storage_mode=="s3"
    }

def build_data_handler(config):
    if config["storage_mode"] == "s3":
        return DataHandler(bucket=config["s3_bucket"], storage_mode="s3")
    else:
        return DataHandler(storage_mode="local")

def print_dataframe_info(label, df):
    """Utility function to print DataFrame shape and columns."""
    logger.info(f"[{label}] shape={df.shape}, columns={list(df.columns)}")

def main():
    config = load_config()
    logger.info(f"[DataProcessingMain] config={config}")
    data_handler = build_data_handler(config)

    # 1) Load aggregated data (price, news)
    ticker = config["ticker"]
    date_range = f"{config['start_date']}_to_{config['end_date']}"
    logger.info(f"Processing for ticker={ticker}, date_range={date_range}")

    def no_op_fetch():
        return pd.DataFrame()

    # The DataHandler callable usage:
    price_df = data_handler(ticker, date_range, "price", no_op_fetch, stage='price')
    news_df = data_handler(ticker, date_range, "news", no_op_fetch, stage='news')

    print_dataframe_info("Loaded price_df", price_df)
    print_dataframe_info("Loaded news_df", news_df)

    if price_df.empty or "DateTime" not in price_df.columns:
        logger.error("'DateTime' column missing or price_df empty. Check your aggregator output.")
        return
    if news_df.empty or "time_published" not in news_df.columns:
        logger.error("'time_published' column missing or news_df empty. Check your aggregator output.")
        return

    # 2) Data Processing
    processor = DataProcessor(price_df, news_df)

    # Generate time horizon combos (for later usage)
    thm = TimeHorizonManager()
    combos = thm.generate_horizon_combos()
    logger.info(f"Generated {len(combos)} combos, e.g. {combos[:5]}...")

    # We define a large set to extract the max gather for MarketAnalyzer
    # Example approach: gather all possible minute intervals
    times = [int(c["gather_td"].total_seconds() // 60) for c in combos]
    max_gather_minutes = max(times) if times else 120  # fallback if combos empty

    # 3) Full pipeline (clean price, merge, sentiment, embedding, etc.)
    # Note the pipeline calls DataProcessor's internal steps.
    # process_pipeline expects time_horizons with gather/predict "time_horizon" keys
    # but we only need the largest gather for the step in MarketAnalyzer below.

    # In your code, we pass a list of dicts with "time_horizon" to process_pipeline.
    time_horizons = [
        {"target_name": f"{m}_minutes", "time_horizon": timedelta(minutes=m)} for m in range(5, max_gather_minutes+1, 5)
    ]
    processed_df = processor.process_pipeline(time_horizons)

    print_dataframe_info("Final processed DataFrame", processed_df)

    # 4) Save embeddings if generated
    if processor.embeddings is not None:
        embeddings_filename = f"{ticker}_embeddings_{date_range}.npy"
        processor.embedder.save_embeddings(processor.embeddings, embeddings_filename, data_handler, stage='embeddings')

    # 5) Save final preprocessed DataFrame
    def fetch_preprocessed():
        return processed_df

    _ = data_handler(ticker, date_range, "preprocessed", fetch_preprocessed, stage='preprocessed')
    logger.info("[DataProcessingMain] Saved final preprocessed data.")

if __name__ == "__main__":
    main()
