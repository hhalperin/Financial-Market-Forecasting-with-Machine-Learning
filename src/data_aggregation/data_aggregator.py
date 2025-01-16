from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import time
from .stock_price_data_gatherer import StockPriceDataGatherer
from .news_data_gatherer import NewsDataGatherer
from utils.logger import get_logger

logger = get_logger('DataAggregator')

class DataAggregator:
    def __init__(self, ticker, start_date, end_date, interval='1min', outputsize='full', data_handler=None, local_mode=False):
        self.ticker = ticker
        self.start_date, self.end_date = self._validate_date(start_date, end_date)
        self.interval = interval
        self.outputsize = outputsize
        self.data_handler = data_handler
        self.local_mode = local_mode

    def _validate_date(self, start_str, end_str):
        def correct_date(ds):
            try:
                return datetime.strptime(ds, "%Y-%m-%d")
            except ValueError:
                year, month, day = map(int, ds.split('-'))
                last_valid = (datetime(year, month, 1) + relativedelta(months=1) - relativedelta(days=1)).day
                return datetime(year, month, min(day, last_valid))

        s_date = correct_date(start_str)
        e_date = correct_date(end_str)
        return s_date.strftime("%Y-%m-%d"), e_date.strftime("%Y-%m-%d")

    def aggregate_data(self):
        total_start = time.time()
        price_df = self._fetch_price_data()
        news_df = self._fetch_news_data()
        logger.info(f"Data aggregation completed in {time.time() - total_start:.2f}s.")
        return price_df, news_df

    def _fetch_price_data(self):
        gatherer = StockPriceDataGatherer(
            ticker=self.ticker,
            interval=self.interval,
            outputsize=self.outputsize,
            local_mode=self.local_mode
        )
        return gatherer.run()

    def _fetch_news_data(self):
        gatherer = NewsDataGatherer(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            local_mode=self.local_mode
        )
        return gatherer.run()
