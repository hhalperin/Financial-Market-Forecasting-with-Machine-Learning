"""
Data Aggregator Module

Aggregates intraday stock price data and news articles for a given ticker and date range.
Uses threading to fetch data concurrently for improved performance.
"""

import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
from src.utils.logger import get_logger
from src.data_aggregation.stock_price_data_gatherer import StockPriceDataGatherer
from src.data_aggregation.news_data_gatherer import NewsDataGatherer

logger = get_logger("DataAggregator")


class DataAggregator:
    """
    Aggregates price and news data concurrently for a specified stock ticker and date range.
    """

    def __init__(self, ticker: str, start_date: str, end_date: str,
                 interval: str = "1min", data_handler=None, local_mode: bool = False) -> None:
        """
        Initializes the DataAggregator with parameters for data retrieval.

        :param ticker: Stock ticker symbol.
        :param start_date: Start date in 'YYYY-MM-DD' format.
        :param end_date: End date in 'YYYY-MM-DD' format.
        :param interval: Interval for intraday data.
        :param data_handler: Optional data handler for saving or further processing.
        :param local_mode: Flag to indicate if the application runs in local mode.
        """
        self.ticker: str = ticker
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.interval: str = interval
        self.data_handler = data_handler
        self.local_mode: bool = local_mode

    def _fetch_price_data(self) -> pd.DataFrame:
        """
        Fetches intraday price data using StockPriceDataGatherer.

        :return: DataFrame containing stock price data.
        """
        gatherer = StockPriceDataGatherer(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            interval=self.interval,
            local_mode=self.local_mode
        )
        return gatherer.run()

    def _fetch_news_data(self) -> pd.DataFrame:
        """
        Fetches news data using NewsDataGatherer.

        :return: DataFrame containing news articles.
        """
        from src.data_aggregation.news_data_gatherer import NewsDataGatherer
        gatherer = NewsDataGatherer(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            local_mode=self.local_mode
        )
        return gatherer.run()

    def aggregate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Concurrently fetches price and news data.
        (Future improvement: Consider asynchronous I/O if API calls become rate limited.)

        :return: Tuple containing (price_data, news_data) DataFrames.
        """
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_price = executor.submit(self._fetch_price_data)
            future_news = executor.submit(self._fetch_news_data)
            price_df = future_price.result()
            news_df = future_news.result()
        elapsed = time.time() - start_time
        logger.info(f"Data aggregation completed in {elapsed:.2f} seconds.")
        return price_df, news_df
