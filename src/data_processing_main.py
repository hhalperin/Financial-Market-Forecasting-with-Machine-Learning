# src/data_processing_main.py

import os
from dotenv import load_dotenv
import pandas as pd
from utils.logger import get_logger
from utils.data_handler import DataHandler
from data_processing.data_processing import DataProcessor
from data_processing.time_horizon_manager import TimeHorizonManager
from data_processing.data_embedder import DataEmbedder
from datetime import timedelta

logger = get_logger("DataProcessingMain")

def print_dataframe_info(label, df):
    """Utility function to print DataFrame shape and columns."""
    logger.info(f"[{label}] shape={df.shape}, columns={list(df.columns)}")

def main():
    load_dotenv()

    # 1) Determine local vs. cloud
    local_mode = os.getenv("LOCAL_MODE", "false").lower() == "true"
    storage_mode = "local" if local_mode else "s3"
    logger.info(f"[DataProcessingMain] local_mode={local_mode}, storage_mode={storage_mode}")

    # 2) Build DataHandler
    data_handler = DataHandler(
        base_data_dir='../data', storage_mode=storage_mode, bucket=os.getenv("S3_BUCKET")
    )

    # 3) Load aggregated data (price, news)
    ticker = "AAPL"
    date_range = "2023-01-01_to_2024-01-31"

    def no_op_fetch():
        return pd.DataFrame()

    price_df = data_handler(ticker, date_range, "price", no_op_fetch, stage='aggregated')
    news_df = data_handler(ticker, date_range, "news", no_op_fetch, stage='aggregated')

    print_dataframe_info("Loaded price_df", price_df)
    print_dataframe_info("Loaded news_df", news_df)

    # Ensure required columns exist
    if 'DateTime' not in price_df.columns:
        logger.error("'DateTime' column missing in price_df. Check your input data.")
        return
    if 'time_published' not in news_df.columns:
        logger.error("'time_published' column missing in news_df. Check your input data.")
        return

    # 4) Data Processing
    processor = DataProcessor(price_df, news_df)

    # Generate time horizon configurations
    thm = TimeHorizonManager()
    testing = True
    combos = thm.testing_horizon_combos(max_combos=10) if testing else thm.generate_horizon_combos(max_combos=1500)

    # Ensure time horizons are in timedelta format
    for combo in combos:
        combo["gather_td"] = timedelta(minutes=int(combo["gather_td"].total_seconds() // 60))
        combo["predict_td"] = timedelta(minutes=int(combo["predict_td"].total_seconds() // 60))

    logger.info(f"Generated {len(combos)} time horizon combos.")

    time_horizons = [
        {"target_name": combo["gather_name"], "time_horizon": combo["gather_td"]} for combo in combos
    ] + [
        {"target_name": combo["predict_name"], "time_horizon": combo["predict_td"]} for combo in combos
    ]

    target_configs = [
        {"time_horizon": combo["predict_td"], "target_name": f"target_{combo['predict_name']}"} for combo in combos
    ]

    # Preprocess and process pipeline
    processed_df = processor.process_pipeline(time_horizons, target_configs)
    print_dataframe_info("Final processed DataFrame", processed_df)

    # Save embeddings (if generated)
    if processor.embeddings is not None:
        embeddings_filename = f"{ticker}_embeddings_{date_range}.npy"
        processor.embedder.save_embeddings(processor.embeddings, embeddings_filename, data_handler, stage='embeddings')


    # 6) Save final preprocessed data
    def fetch_preprocessed():
        return processed_df

    processed_data = data_handler(
        ticker, date_range, "preprocessed", fetch_preprocessed, stage='preprocessed'
    )
    logger.info(f"Saved final preprocessed data: shape={processed_data.shape}")

if __name__ == "__main__":
    main()
