import requests
import pandas as pd
import time
from config import Config
from logger import get_logger
from error_handler import handle_api_errors
from utils.data_handler import DataHandler

class StockPriceDataGatherer:
    """
    Responsible for fetching stock price data and technical indicators from AlphaVantage API.
    """

    def __init__(self, ticker, interval='5min', outputsize='full', month=None):
        self.ticker = ticker
        self.interval = interval
        self.outputsize = outputsize
        self.api_key = Config.ALPHAVANTAGE_API_KEY
        self.base_url = 'https://www.alphavantage.co/query'
        self.logger = get_logger(self.__class__.__name__)
        self.data_handler = DataHandler()  # Use the DataHandler for data loading/saving
        self.month = month  # Accept a specific month for querying data
        self.rate_limit_pause = 12  # Pause time to avoid rate limiting issues

    @handle_api_errors
    def fetch_intraday_data(self):
        """
        Fetches intraday stock price data for a specific month if provided.
        """
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': self.ticker,
            'interval': self.interval,
            'outputsize': self.outputsize,
            'apikey': self.api_key
        }

        # Include the 'month' parameter if provided to query historical data for that specific month
        if self.month:
            params['month'] = self.month

        response = requests.get(self.base_url, params=params)

        # Log the full URL for debugging
        full_url = response.url
        self.logger.debug(f"API Request URL: {full_url}")

        data = response.json()

        key = f'Time Series ({self.interval})'
        if key in data:
            time_series_data = data[key]
            df = pd.DataFrame.from_dict(time_series_data, orient='index')
            df = self._process_price_dataframe(df)
            self.logger.info(f"Fetched intraday data for {self.ticker} (Month: {self.month})")
            return df
        else:
            self.logger.error(f"API Response: {data}")
            error_msg = data.get('Note') or data.get('Error Message') or data.get('Information') or 'Unknown error occurred.'
            self.logger.error(f"Error fetching intraday data: {error_msg}. Full URL: {full_url}")
            return pd.DataFrame()  # Return an empty DataFrame instead of raising an error

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

        # Add additional parameters based on the indicator type
        if indicator == 'RSI':
            params.update({
                'time_period': 14,  # Default time period for RSI
                'series_type': 'close'  # Calculate RSI based on close prices
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
        self.logger.debug(f"API Request URL for {indicator}: {full_url}")

        data = response.json()

        key = f'Technical Analysis: {indicator}'
        if key in data:
            technical_data = data[key]
            df = pd.DataFrame.from_dict(technical_data, orient='index')
            df = self._process_technical_dataframe(df, indicator)
            self.logger.info(f"Fetched {indicator} data for {self.ticker} (Month: {self.month})")
            return df
        else:
            self.logger.error(f"API Response: {data}")
            error_msg = data.get('Note') or data.get('Error Message') or data.get('Information') or 'Unknown error occurred.'
            self.logger.error(f"Error fetching {indicator} data: {error_msg}. Full URL: {full_url}")
            return pd.DataFrame()  # Return an empty DataFrame instead of raising an error

    def _process_price_dataframe(self, df):
        df.rename(columns={
            '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
            '4. close': 'Close', '5. volume': 'Volume'}, inplace=True)
        df['Symbol'] = self.ticker
        df.index = pd.to_datetime(df.index)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'DateTime'}, inplace=True)
        return df[['Symbol', 'DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']]

    def _process_technical_dataframe(self, df, indicator):
        """
        Processes the technical indicator DataFrame by renaming columns and converting the index.
        """
        df.index = pd.to_datetime(df.index)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'DateTime'}, inplace=True)
        new_column_names = {col: f"{indicator}_{col}" for col in df.columns if col != 'DateTime'}
        df.rename(columns=new_column_names, inplace=True)
        return df

    def run(self):
        """
        Orchestrates fetching of price data and technical indicators.
        """
        # Fetching price data
        price_df = self.fetch_intraday_data()
        if price_df.empty:
            self.logger.error(f"No price data available for {self.ticker} for the month: {self.month}")
            return price_df  # Early return if no data fetched

        time.sleep(self.rate_limit_pause)  # To avoid rate limiting

        # Fetch RSI and MACD indicators
        rsi_df = self.fetch_technical_indicator('RSI')
        time.sleep(self.rate_limit_pause)  # To avoid rate limiting
        macd_df = self.fetch_technical_indicator('MACD')
        time.sleep(self.rate_limit_pause)  # To avoid rate limiting

        # Merge the price data with RSI and MACD dataframes
        merged_df = price_df
        if not rsi_df.empty:
            merged_df = merged_df.merge(rsi_df, on='DateTime', how='left')
        else:
            self.logger.warning(f"No RSI data available for {self.ticker} (Month: {self.month})")

        if not macd_df.empty:
            merged_df = merged_df.merge(macd_df, on='DateTime', how='left')
        else:
            self.logger.warning(f"No MACD data available for {self.ticker} (Month: {self.month})")

        self.logger.info(f"Merged price data with technical indicators for {self.ticker} (Month: {self.month})")

        # Saving the merged DataFrame to CSV for inspection
        merged_df.to_csv(f'{self.ticker}_merged_data_{self.month}.csv', index=False)
        return merged_df
