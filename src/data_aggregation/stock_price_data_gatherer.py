"""
Stock Price Data Gatherer Module

Gathers intraday stock price data from Alpha Vantage by splitting the overall date range
into monthly chunks. This helps manage API limitations and large datasets. For improved I/O,
consider switching from CSV to Parquet when storing large datasets.
"""

import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List
from src.data_aggregation.base_data_gatherer import BaseDataGatherer
from src.utils.logger import get_logger

class StockPriceDataGatherer(BaseDataGatherer):
    """
    Gathers intraday stock price data for a given ticker over a specified date range.
    Splits the API requests into monthly chunks.
    """

    def __init__(self, ticker: str, start_date: str, end_date: str,
                 interval: str = "1min", local_mode: bool = False) -> None:
        """
        Initializes the StockPriceDataGatherer.

        :param ticker: Stock ticker symbol.
        :param start_date: Start date ('YYYY-MM-DD').
        :param end_date: End date ('YYYY-MM-DD').
        :param interval: Data interval (default "1min").
        :param local_mode: Flag for local mode.
        """
        super().__init__(ticker, local_mode=local_mode)
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.interval: str = interval
        self.base_url: str = "https://www.alphavantage.co/query"
        self.logger = get_logger(self.__class__.__name__)

    def _generate_month_params(self) -> List[str]:
        """
        Generates monthly query parameters for each month within the date range.

        :return: List of strings, e.g. ["month=2022-01", "month=2022-02", ...].
        """
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        date_list: List[str] = []
        current_date = start_dt

        while current_date <= end_dt:
            date_str = current_date.strftime("%Y-%m")
            date_list.append(f"month={date_str}")
            current_date += relativedelta(months=1)

        return date_list

    def _fetch_monthly_data(self) -> pd.DataFrame:
        """
        Fetches intraday price data for each month, concatenates the results, and cleans the data.
        (Future improvement: Use chunked processing and switch to Parquet for better performance.)

        :return: Consolidated DataFrame of stock price data.
        """
        date_list = self._generate_month_params()
        df_list = []

        for date_frag in date_list:
            url = (
                f"{self.base_url}?function=TIME_SERIES_INTRADAY"
                f"&symbol={self.ticker}"
                f"&interval={self.interval}"
                f"&{date_frag}"
                f"&outputsize=full"
                f"&apikey={self.api_key}"
            )
            
            #self.logger.info(f"[DEBUG] Fetching stock data with URL: {url}")
            
            try:
                data = self.make_api_request(url)
            except Exception as e:
                self.logger.error(f"[ERROR] API request failed for {self.ticker} - {e}")
                continue

            #self.logger.info(f"[DEBUG] API Response for {self.ticker} on {date_frag}: {data}")

            if isinstance(data, dict):
                if "Note" in data:
                    self.logger.error(f"[ERROR] API Limit reached: {data['Note']}")
                    return pd.DataFrame()
                if "Error Message" in data:
                    self.logger.error(f"[ERROR] API Error: {data['Error Message']}")
                    return pd.DataFrame()
                if f"Time Series ({self.interval})" not in data:
                    self.logger.warning(f"[WARNING] No data found for {self.ticker} on {date_frag}: {data}")
                    continue

            ts_data = data[f"Time Series ({self.interval})"]
            df = pd.DataFrame.from_dict(ts_data, orient="index")

            # Rename columns to a standard format.
            df.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume"
            }, inplace=True)

            df["Symbol"] = self.ticker
            df.reset_index(inplace=True)
            df.rename(columns={"index": "DateTime"}, inplace=True)
            df_list.append(df)

        if not df_list:
            self.logger.error(f"[ERROR] No valid data retrieved for {self.ticker}. API may be rate limiting.")
            return pd.DataFrame()

        pricing_df = pd.concat(df_list, ignore_index=True)
        # Reorder columns to a standard order.
        pricing_df = pricing_df[["Symbol", "DateTime", "Open", "High", "Low", "Close", "Volume"]]
        pricing_df["DateTime"] = pd.to_datetime(pricing_df["DateTime"], errors="coerce")
        pricing_df.sort_values("DateTime", inplace=True)
        pricing_df.drop_duplicates(subset=["DateTime"], inplace=True)
        pricing_df.reset_index(drop=True, inplace=True)

        return pricing_df

    def run(self) -> pd.DataFrame:
        """
        Main entry point for fetching intraday stock price data.

        :return: Processed DataFrame with stock price data.
        """
        return self._fetch_monthly_data()
