# src/data_aggregation/data_aggregator.py

import time
import pandas as pd
from datetime import datetime
from src.utils.logger import get_logger
from src.data_aggregation.stock_price_data_gatherer import StockPriceDataGatherer
from src.data_aggregation.news_data_gatherer import NewsDataGatherer

logger = get_logger("DataAggregator")

class DataAggregator:
    """
    Aggregates monthly-chunk intraday price data + news data for a given ticker & date range,
    then merges them on the 'DateTime' columns if desired.
    """

    def __init__(
        self,
        ticker,
        start_date,
        end_date,
        interval='1min',
        data_handler=None,
        local_mode=False
    ):
        """
        :param ticker: e.g. 'AAPL'
        :param start_date: e.g. '2022-01-01'
        :param end_date:   e.g. '2022-06-01'
        :param interval:   e.g. '1min', '5min'
        :param data_handler: not strictly used here, but can be stored if needed
        :param local_mode: if True, use local environment for API key
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.data_handler = data_handler
        self.local_mode = local_mode

    def _fetch_price_data(self):
        """
        Uses StockPriceDataGatherer (monthly approach) to fetch intraday data.
        """
        gatherer = StockPriceDataGatherer(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            interval=self.interval,
            local_mode=self.local_mode
        )
        price_df = gatherer.run()
        return price_df

    def _fetch_news_data(self):
        """
        Uses NewsDataGatherer to fetch up to 1000 articles for [start_date, end_date].
        """
        gatherer = NewsDataGatherer(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            local_mode=self.local_mode
        )
        news_df = gatherer.run()
        return news_df

    def aggregate_data(self):
        """
        Main method: fetch monthly-chunk intraday + news, then return both.
        """
        start_time = time.time()
        price_df = self._fetch_price_data()
        news_df = self._fetch_news_data()
        elapsed = time.time() - start_time
        logger.info(f"Data aggregation completed in {elapsed:.2f}s.")
        return price_df, news_df
