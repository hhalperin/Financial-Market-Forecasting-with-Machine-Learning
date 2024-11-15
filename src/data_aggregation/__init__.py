import os
import time
import pandas as pd
from datetime import datetime, timedelta
from logger import get_logger
from .news_data_gatherer import NewsDataGatherer
from .stock_price_data_gatherer import StockPriceDataGatherer
from utils.data_handler import DataHandler
from dateutil.relativedelta import relativedelta

logger = get_logger('DataAggregation')

class DataAggregator:
    def __init__(self, ticker: str, start_date: str, end_date: str, interval: str = '1min', outputsize: str = 'full', data_handler=None):
        """
        Initialize the DataAggregator class with the necessary parameters.
        """
        self.ticker = ticker
        self.start_date, self.end_date = self.validate_date(start_date, end_date)
        self.interval = interval
        self.outputsize = outputsize
        self.data_handler = data_handler if data_handler else DataHandler()
        self.logger = logger

    def validate_date(self, start_date_str, end_date_str):
        """
        Validate and correct start and end dates if necessary.
        """
        def correct_date(date_str):
            try:
                return datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                year, month, day = map(int, date_str.split('-'))
                last_valid_day = (datetime(year, month, 1) + relativedelta(months=1) - relativedelta(days=1)).day
                corrected_day = min(day, last_valid_day)
                return datetime(year, month, corrected_day)

        corrected_start_date = correct_date(start_date_str).strftime('%Y-%m-%d')
        end_date_dt = correct_date(end_date_str)
        last_valid_day = (datetime(end_date_dt.year, end_date_dt.month, 1) + relativedelta(months=1) - relativedelta(days=1)).day
        corrected_end_date = datetime(end_date_dt.year, end_date_dt.month, last_valid_day).strftime('%Y-%m-%d')

        return corrected_start_date, corrected_end_date

    def fetch_price_data(self):
        """
        Fetch price data for the entire date range.
        """
        date_range = f"{self.start_date}_to_{self.end_date}"
        price_df = self.data_handler(
            ticker=self.ticker,
            date_range=date_range,
            data_type="price",
            data_fetcher=lambda: self._fetch_price_for_range()
        )

        if price_df is None or price_df.empty:
            self.logger.error(f"No price data was fetched for the entire date range: {self.start_date} to {self.end_date}.")
        else:
            self.logger.info(f"Fetched price data with shape: {price_df.shape}")

        return price_df

    def _fetch_price_for_range(self):
        """
        Fetch price data for the entire range using the StockPriceDataGatherer.
        """
        monthly_date_ranges = self.generate_monthly_date_ranges()
        all_price_data = []

        for month in monthly_date_ranges:
            self.logger.info(f"Fetching price data for {self.ticker} for the month: {month}...")
            price_gatherer = StockPriceDataGatherer(ticker=self.ticker, interval=self.interval, outputsize=self.outputsize, month=month)
            monthly_price_df = price_gatherer.run()

            if monthly_price_df is not None and not monthly_price_df.empty:
                all_price_data.append(monthly_price_df)

        if all_price_data:
            price_df = pd.concat(all_price_data, ignore_index=True)
            self.logger.info(f"Combined price data shape: {price_df.shape}")
        else:
            price_df = pd.DataFrame()
            self.logger.error(f"No price data was fetched for the entire date range: {self.start_date} to {self.end_date}.")

        return price_df

    def fetch_news_data(self):
        """
        Fetch news data for the given ticker and date range.
        """
        date_range = f"{self.start_date}_to_{self.end_date}"
        news_df = self.data_handler(
            ticker=self.ticker,
            date_range=date_range,
            data_type="news",
            data_fetcher=lambda: self._fetch_news()
        )

        if news_df is None or news_df.empty:
            self.logger.error(f"No news data was fetched for the entire date range: {self.start_date} to {self.end_date}.")
        else:
            self.logger.info(f"Fetched news data with shape: {news_df.shape}")

        return news_df

    def _fetch_news(self):
        """
        Fetch news data using NewsDataGatherer.
        """
        self.logger.info(f"Fetching news data for {self.ticker} from {self.start_date} to {self.end_date}...")
        news_gatherer = NewsDataGatherer(ticker=self.ticker, start_date=self.start_date, end_date=self.end_date)
        return news_gatherer.run()

    def generate_monthly_date_ranges(self) -> list:
        """
        Generates a list of monthly date ranges between start_date and end_date.
        """
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")

        current_date = start_date
        date_ranges = []

        while current_date <= end_date:
            month_str = current_date.strftime("%Y-%m")
            date_ranges.append(month_str)
            next_month = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1)
            current_date = next_month

        return date_ranges

    def aggregate_data(self):
        """
        Orchestrates the entire data aggregation process, fetching price and news data.
        """
        total_start_time = time.time()

        # Fetch price data
        price_df = self.fetch_price_data()

        # Fetch news data
        news_df = self.fetch_news_data()

        # Log completion time
        aggregation_end_time = time.time()
        self.logger.info(f"Data aggregation completed in {aggregation_end_time - total_start_time:.2f} seconds.")

        return price_df, news_df
