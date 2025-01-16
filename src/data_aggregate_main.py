import os
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.data_handler import DataHandler
from data_aggregation.data_aggregator import DataAggregator

logger = get_logger("DataAggregationMain")

def load_config():
    """
    Load and validate environment variables.
    """
    load_dotenv()
    local_mode = os.getenv("LOCAL_MODE", "false").lower() == "true"
    storage_mode = "local" if local_mode else "s3"
    s3_bucket = os.getenv("S3_BUCKET") if not local_mode else None

    if not local_mode and not s3_bucket:
        raise ValueError("S3_BUCKET environment variable must be set when LOCAL_MODE is false.")

    ticker = os.getenv("TICKER", "AAPL")
    start_date = os.getenv("START_DATE", "2023-01-01")
    end_date = os.getenv("END_DATE", "2024-01-31")
    interval = os.getenv("INTERVAL", "1min")
    outputsize = os.getenv("OUTPUTSIZE", "full")

    return {
        "local_mode": local_mode,
        "storage_mode": storage_mode,
        "s3_bucket": s3_bucket,
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval,
        "outputsize": outputsize,
    }

def setup_data_handler(config):
    """
    Set up the appropriate DataHandler based on the configuration.
    """
    if config["local_mode"]:
        logger.info("[SETUP] Using local file storage.")
        return DataHandler(base_data_dir="../data", storage_mode="local")
    else:
        logger.info(f"[SETUP] Using S3 file storage with bucket: {config['s3_bucket']}")
        return DataHandler(bucket=config["s3_bucket"], storage_mode="s3")

def setup_aggregator(config, data_handler):
    """
    Create and return a DataAggregator instance.
    """
    return DataAggregator(
        ticker=config["ticker"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        interval=config["interval"],
        outputsize=config["outputsize"],
        data_handler=data_handler,
        local_mode=config["local_mode"],
    )

def main():
    # Load configuration
    logger.info("[MAIN] Loading configuration...")
    config = load_config()
    logger.info(f"[MAIN] Configuration loaded: {config}")

    # Set up DataHandler and DataAggregator
    data_handler = setup_data_handler(config)
    aggregator = setup_aggregator(config, data_handler)

    # Perform data aggregation
    logger.info("[MAIN] Starting data aggregation...")
    price_df, news_df = aggregator.aggregate_data()

    # Save aggregated data
    if price_df is not None and not price_df.empty:
        logger.info("[MAIN] Saving price data...")
        data_handler.save_data(price_df, f"{config['ticker']}_price_{config['start_date']}_to_{config['end_date']}.csv", "price", "aggregated")
    else:
        logger.warning("[MAIN] No price data was aggregated.")

    if news_df is not None and not news_df.empty:
        logger.info("[MAIN] Saving news data...")
        data_handler.save_data(news_df, f"{config['ticker']}_news_{config['start_date']}_to_{config['end_date']}.csv", "news", "aggregated")
    else:
        logger.warning("[MAIN] No news data was aggregated.")

    logger.info("[MAIN] Data aggregation process completed.")


if __name__ == "__main__":
    main()
