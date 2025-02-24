# src/trading/__init__.py
"""
Trading Module

This module contains components for live trading simulation:
- trading_config: Trading-specific configuration.
- live_data_collector: Continuously collects live data.
- trading_engine: Orchestrates processing, prediction, and trade decision making.
- trading_rules: Contains the trading decision logic.
- trade_simulator: Simulates trade executions and logs results.
- dashboard: Provides a dashboard interface (future module).
"""

from .trading_config import trading_settings
from .live_data_collector import LiveDataCollector
from .trading_engine import TradingEngine
from .trading_rules import TradingRules
from .trade_simulator import TradeSimulator
from .dashboard import Dashboard
