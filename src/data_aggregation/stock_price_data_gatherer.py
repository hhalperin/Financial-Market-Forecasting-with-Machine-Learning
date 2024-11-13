import requests
import pandas as pd
import time
import functools
from config import Config
from logger import get_logger
from error_handler import handle_api_errors
from utils.data_handler import DataHandler

class StockPriceDataGatherer:
    """
    Responsible for fetching stock price data and technical indicators from AlphaVantage API.
    """

    def __init__(self, ticker, interval='5min', outputsize='full'):
        self.ticker = ticker
        self.interval = interval
        self.outputsize = outputsize
        self.api_key = Config.ALPHAVANTAGE_API_KEY
        self.base_url = 'https://www.alphavantage.co/query'
        self.logger = get_logger(self.__class__.__name__)
        self.data_handler = DataHandler()  # Use the DataHandler for data loading/saving
        self.rate_limit_pause = 0  # Pause time to avoid rate limiting issues

    @handle_api_errors
    def fetch_intraday_data(self):
        """
        Fetches intraday stock price data.
        """
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': self.ticker,
            'interval': self.interval,
            'outputsize': self.outputsize,
            'apikey': self.api_key
        }
        response = requests.get(self.base_url, params=params)

        # Log the full URL for debugging
        full_url = response.url
        #self.logger.debug(f"API Request URL: {full_url}")

        data = response.json()

        key = f'Time Series ({self.interval})'
        if key in data:
            time_series_data = data[key]
            df = pd.DataFrame.from_dict(time_series_data, orient='index')
            df = self._process_price_dataframe(df)
            self.logger.info(f"Fetched intraday data for {self.ticker}")
            return df
        else:
            self.logger.error(f"API Response: {data}")
            error_msg = data.get('Note') or data.get('Error Message') or data.get('Information') or 'Unknown error occurred.'
            self.logger.error(f"Error fetching intraday data: {error_msg}. Full URL: {full_url}")
            raise ValueError(f"Error fetching data: {error_msg}", full_url)

    @handle_api_errors
    def fetch_technical_indicator(self, indicator):
        """
        Fetches technical indicators like RSI and MACD.
        """
        params = {
            'function': indicator,
            'symbol': self.ticker,
            'interval': self.interval,
            'apikey': self.api_key
        }

        if indicator == 'RSI':
            params.update({
                'time_period': 14,
                'series_type': 'close'
            })
        elif indicator == 'MACD':
            params.update({
                'series_type': 'close',
                'fastperiod': 12,
                'slowperiod': 26,
                'signalperiod': 9
            })

        response = requests.get(self.base_url, params=params)

        # Log the full URL for debugging
        full_url = response.url
        #self.logger.debug(f"API Request URL for {indicator}: {full_url}")

        data = response.json()

        key = f'Technical Analysis: {indicator}'
        if key in data:
            technical_data = data[key]
            df = pd.DataFrame.from_dict(technical_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'DateTime'}, inplace=True)
            self.logger.info(f"Fetched {indicator} data for {self.ticker}")
            return df
        else:
            self.logger.error(f"API Response: {data}")
            error_msg = data.get('Note') or data.get('Error Message') or data.get('Information') or 'Unknown error occurred.'
            #self.logger.error(f"Error fetching {indicator} data: {error_msg}. Full URL: {full_url}")
            raise ValueError(f"Error fetching data: {error_msg}", full_url)

    def _process_price_dataframe(self, df):
        df.rename(columns={
            '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
            '4. close': 'Close', '5. volume': 'Volume'}, inplace=True)
        df['Symbol'] = self.ticker
        df.index = pd.to_datetime(df.index)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'DateTime'}, inplace=True)
        return df[['Symbol', 'DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']]

    def run(self):
        """
        Orchestrates fetching of price data and technical indicators.
        """
        # Fetching data
        price_df = self.fetch_intraday_data()
        time.sleep(self.rate_limit_pause)  # To avoid rate limiting

        # Fetch RSI and MACD indicators
        rsi_df = self.fetch_technical_indicator('RSI')
        time.sleep(self.rate_limit_pause)  # To avoid rate limiting
        macd_df = self.fetch_technical_indicator('MACD')
        time.sleep(self.rate_limit_pause)  # To avoid rate limiting

        # Merging price and indicator data
        merged_df = price_df.merge(rsi_df, on='DateTime', how='left')
        merged_df = merged_df.merge(macd_df, on='DateTime', how='left')

        self.logger.info(f"Merged price data with technical indicators for {self.ticker}")
        return merged_df
