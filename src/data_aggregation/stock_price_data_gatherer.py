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
        """
        Initializes the StockPriceDataGatherer with the required parameters.
        
        Args:
            ticker (str): The stock ticker symbol.
            interval (str): The interval for fetching stock price data ('1min', '5min', etc.).
            outputsize (str): The amount of data ('compact' for recent data, 'full' for entire dataset).
            month (str): Specific month to fetch data for, in 'YYYY-MM' format.
        """
        self.ticker = ticker
        self.interval = interval
        self.outputsize = outputsize
        self.api_key = Config.ALPHAVANTAGE_API_KEY
        self.base_url = 'https://www.alphavantage.co/query'
        self.logger = get_logger(self.__class__.__name__)
        self.data_handler = DataHandler()  # Instance of DataHandler for saving/loading data.
        self.month = month  # Month parameter used for querying specific data.
        self.rate_limit_pause = 0  # Pause time to avoid API rate limiting.

    @handle_api_errors
    def fetch_intraday_data(self):
        """
        Fetches intraday stock price data from the AlphaVantage API.

        Returns:
            pd.DataFrame: A DataFrame containing the fetched price data.
        """
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': self.ticker,
            'interval': self.interval,
            'outputsize': self.outputsize,
            'apikey': self.api_key
        }

        # Include the 'month' parameter if provided to query historical data for a specific month.
        if self.month:
            params['month'] = self.month

        # Make a request to the API.
        response = requests.get(self.base_url, params=params)
        data = response.json()

        # Check if the expected key is in the response.
        key = f'Time Series ({self.interval})'
        if key in data:
            # Convert the time series data to a DataFrame.
            time_series_data = data[key]
            df = pd.DataFrame.from_dict(time_series_data, orient='index')
            return self._process_price_dataframe(df)
        else:
            # Log an error if the data is not available or an error occurred.
            error_msg = data.get('Note') or data.get('Error Message') or data.get('Information') or 'Unknown error occurred.'
            self.logger.error(f"Error fetching intraday data: {error_msg}.")
            return pd.DataFrame()  # Return an empty DataFrame instead of raising an error.

    @handle_api_errors
    def fetch_technical_indicator(self, indicator):
        """
        Fetches a technical indicator (e.g., RSI or MACD) for the stock.

        Args:
            indicator (str): The technical indicator to fetch (e.g., 'RSI', 'MACD').

        Returns:
            pd.DataFrame: A DataFrame containing the technical indicator data.
        """
        params = {
            'function': indicator,
            'symbol': self.ticker,
            'interval': self.interval,
            'apikey': self.api_key
        }

        # Add additional parameters depending on the indicator type.
        if indicator == 'RSI':
            params.update({
                'time_period': 14,  # Default time period for RSI.
                'series_type': 'close'  # RSI calculated based on close prices.
            })
        elif indicator == 'MACD':
            params.update({
                'series_type': 'close',
                'fastperiod': 12,
                'slowperiod': 26,
                'signalperiod': 9
            })

        # Make a request to the API.
        response = requests.get(self.base_url, params=params)
        data = response.json()

        # Check if the expected key is in the response.
        key = f'Technical Analysis: {indicator}'
        if key in data:
            technical_data = data[key]
            df = pd.DataFrame.from_dict(technical_data, orient='index')
            return self._process_technical_dataframe(df, indicator)
        else:
            # Log an error if the data is not available or an error occurred.
            error_msg = data.get('Note') or data.get('Error Message') or data.get('Information') or 'Unknown error occurred.'
            self.logger.error(f"Error fetching {indicator} data: {error_msg}.")
            return pd.DataFrame()  # Return an empty DataFrame instead of raising an error.

    def _process_price_dataframe(self, df):
        """
        Processes the price data DataFrame by renaming columns and converting the index.

        Args:
            df (pd.DataFrame): The raw price data DataFrame.

        Returns:
            pd.DataFrame: A cleaned and formatted DataFrame with relevant columns.
        """
        # Rename the columns to more descriptive names.
        df.rename(columns={
            '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
            '4. close': 'Close', '5. volume': 'Volume'}, inplace=True)
        df['Symbol'] = self.ticker
        df.index = pd.to_datetime(df.index)  # Convert index to datetime.
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'DateTime'}, inplace=True)
        return df[['Symbol', 'DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']]

    def _process_technical_dataframe(self, df, indicator):
        """
        Processes the technical indicator DataFrame by renaming columns and converting the index.

        Args:
            df (pd.DataFrame): The raw technical indicator DataFrame.
            indicator (str): The technical indicator name (e.g., 'RSI', 'MACD').

        Returns:
            pd.DataFrame: A cleaned and formatted DataFrame with relevant columns.
        """
        df.index = pd.to_datetime(df.index)  # Convert index to datetime.
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'DateTime'}, inplace=True)
        # Rename columns to include the indicator name.
        new_column_names = {col: f"{indicator}_{col}" for col in df.columns if col != 'DateTime'}
        df.rename(columns=new_column_names, inplace=True)
        return df

    def _fetch_and_merge_technical_indicators(self, price_df):
        """
        Fetches RSI and MACD indicators and merges them with the price data.

        Args:
            price_df (pd.DataFrame): DataFrame containing price data.

        Returns:
            pd.DataFrame: Merged DataFrame with price data and technical indicators.
        """
        # Fetch RSI and MACD indicators.
        rsi_df = self.fetch_technical_indicator('RSI')
        time.sleep(self.rate_limit_pause)  # To avoid rate limiting.
        macd_df = self.fetch_technical_indicator('MACD')
        time.sleep(self.rate_limit_pause)  # To avoid rate limiting.

        # Merge technical indicators with price data.
        merged_df = price_df
        if not rsi_df.empty:
            merged_df = merged_df.merge(rsi_df, on='DateTime', how='left')
        else:
            self.logger.warning(f"No RSI data available for {self.ticker} (Month: {self.month})")

        if not macd_df.empty:
            merged_df = merged_df.merge(macd_df, on='DateTime', how='left')
        else:
            self.logger.warning(f"No MACD data available for {self.ticker} (Month: {self.month})")

        return merged_df

    def run(self):
        """
        Orchestrates the entire process of fetching price data and technical indicators.

        Returns:
            pd.DataFrame: The final merged DataFrame containing price and technical indicator data.
        """
        # Step 1: Fetch price data.
        price_df = self.fetch_intraday_data()
        if price_df.empty:
            self.logger.error(f"No price data available for {self.ticker} for the month: {self.month}")
            return price_df  # Early return if no data fetched.

        time.sleep(self.rate_limit_pause)  # To avoid rate limiting.

        # Step 2: Fetch and merge technical indicators.
        stock_pricing_df = self._fetch_and_merge_technical_indicators(price_df)

        return stock_pricing_df
