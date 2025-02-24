# src/trading/main.py

import os
import datetime
import numpy as np
import pandas as pd
from src.trading.trading_config import trading_settings
from src.utils.logger import get_logger
from src.utils.data_handler import DataHandler
from src.data_aggregation.data_aggregator import DataAggregator
from src.trading.trading_engine import TradingEngine
from src.trading.trade_simulator import TradeSimulator
from src.trading.trade_analysis import (
    load_trade_log,
    compute_equity_curve,
    plot_equity_curve,
    plot_drawdown,
    plot_trade_pnl_histogram,
    plot_trade_pnl_over_time,
    compute_summary_metrics,
    save_summary_table
)

def main() -> None:
    logger = get_logger("TradingMain")
    logger.info("Starting Trading System Main Script.")

    # Define directories for intermediate outputs
    base_dir = trading_settings.data_storage_path
    raw_dir = os.path.join(base_dir, "raw")
    processed_dir = os.path.join(base_dir, "processed")
    permanent_storage_path = trading_settings.permanent_storage_path
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(permanent_storage_path, exist_ok=True)
    
    # Create a DataHandler for local storage
    data_handler = DataHandler(base_data_dir=base_dir, storage_mode="local")
    ticker = trading_settings.ticker
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Define filenames for raw and processed data
    raw_price_filename = f"{ticker}_raw_price_{today_str}.csv"
    raw_news_filename = f"{ticker}_raw_news_{today_str}.csv"
    processed_filename = f"{ticker}_processed_{today_str}.csv"
    numeric_filename = f"{ticker}_numeric_{today_str}.npy"  # Numeric data saved as .npy
    
    raw_price_path = os.path.join(raw_dir, raw_price_filename)
    raw_news_path = os.path.join(raw_dir, raw_news_filename)
    processed_path = os.path.join(processed_dir, processed_filename)
    numeric_path = os.path.join(processed_dir, numeric_filename)
    
    # --- Stage 1: Raw Data Aggregation ---
    if os.path.exists(raw_price_path) and os.path.exists(raw_news_path):
        logger.info(f"Loading raw price data from {raw_price_path}")
        raw_price_df = pd.read_csv(raw_price_path)
        logger.info(f"Loading raw news data from {raw_news_path}")
        raw_news_df = pd.read_csv(raw_news_path)
    else:
        logger.error("Raw data files not found. Please run data aggregation first.")
        return

    # --- Stage 2: Data Processing ---
    if os.path.exists(processed_path) and os.path.exists(numeric_path) and os.path.getsize(numeric_path) > 0:
        logger.info(f"Loading processed data from {processed_path}")
        processed_df = pd.read_csv(processed_path, parse_dates=["DateTime"])
        logger.info(f"Loading numeric data from {numeric_path}")
        numeric_array = np.load(numeric_path, allow_pickle=True)
    else:
        logger.error("Processed/numeric data files not found. Please run data processing first.")
        return

    # --- Stage 3: Trading Cycle ---
    engine = TradingEngine(ticker=ticker, local_mode=True)
    # Pass raw data and cached numeric array to the trading engine
    decision, prediction, rationale = engine.execute_trading_cycle(
        price_df=raw_price_df,
        news_df=raw_news_df,
        processed_array=numeric_array
    )
    logger.info(f"Trading decision: {decision}, Prediction: {prediction:.4f}%, Rationale: {rationale}")

if __name__ == "__main__":
    main()
