"""
Data Aggregation Module

This module provides classes to aggregate stock price data and news data.
It serves as the package initializer for the data_aggregation module.
"""

from .data_aggregator import DataAggregator
from .news_data_gatherer import NewsDataGatherer
from .stock_price_data_gatherer import StockPriceDataGatherer
from .base_data_gatherer import BaseDataGatherer

__all__ = [
    "DataAggregator",
    "NewsDataGatherer",
    "BaseDataGatherer",
    "StockPriceDataGatherer"
]
