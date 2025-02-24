# trading_config.py
"""
Trading Configuration Module

This module defines the TradingConfig class which extends the centralized configuration
from src/config.py. It allows trading-specific overrides (e.g., ticker, date range) and
adds parameters for simulation/live mode, Interactive Brokers integration, and additional
machine learning settings for signal aggregation and risk management.
"""

from src.config import Settings  # Base configuration
from pydantic_settings import BaseSettings
from typing import Optional

class TradingConfig(Settings):
    # Trading mode: True for simulation, False for live trading.
    simulation_mode: bool = True

    # Interactive Brokers (IB) API parameters.
    ib_api_host: str = "127.0.0.1"
    ib_api_port: int = 7497
    ib_client_id: int = 1

    # Parameters for aggregating predictions from multiple models.
    # For example, if you have 5-10 best models for different time horizons.
    number_of_best_models: int = 5
    # Weight decay factor can be used to give more weight to the best models and recent data.
    model_weight_decay: float = 0.8

    # Flag to enable the second ML model for computing optimal stop loss and take profit.
    stop_loss_model_enabled: bool = True

    # Optionally override base configuration values for trading.
    ticker: str = "HIMS"            # Trading ticker override.
    start_date: str = "2025-01-01"    # Trading data start date.
    end_date: str = "2025-02-01"      # Trading data end date.

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global trading configuration instance.
trading_config = TradingConfig()
