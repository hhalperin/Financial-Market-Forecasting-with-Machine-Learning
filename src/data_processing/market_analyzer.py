import pandas as pd
import numpy as np
import talib
from src.utils.logger import get_logger


class MarketAnalyzer:
    """
    Handles market-related calculations, including:
    - Price fluctuations over all specified time intervals.
    - Technical indicators like RSI, MACD, and rate of change (ROC).
    """

    def __init__(self, data_df):
        """
        :param data_df: A DataFrame containing stock price data with a 'Close' column.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.data_df = data_df.copy()

    def calculate_price_fluctuations(self, max_gather_minutes, step=5):
        """
        Calculate price changes and percentage changes for intervals up to `max_gather_minutes` in steps of `step`.
        """
        if 'DateTime' not in self.data_df.columns or 'Close' not in self.data_df.columns:
            self.logger.error("Required columns 'DateTime' or 'Close' are missing.")
            return self.data_df

        self.data_df.set_index('DateTime', inplace=True, drop=False)
        self.data_df['Close'] = pd.to_numeric(self.data_df['Close'], errors='coerce')

        # Prepare a dictionary to hold the new columns
        new_columns = {}

        for minutes in range(step, max_gather_minutes + 1, step):
            self.logger.info(f"Calculating price fluctuation for {minutes} minutes.")
            shifted_close = self.data_df['Close'].shift(-minutes)
            new_columns[f"{minutes}_minutes_change"] = shifted_close - self.data_df['Close']
            new_columns[f"{minutes}_minutes_percentage_change"] = (
                (shifted_close - self.data_df['Close']) / self.data_df['Close'] * 100
            )

        # Add all new columns to the DataFrame at once
        self.data_df = pd.concat([self.data_df, pd.DataFrame(new_columns, index=self.data_df.index)], axis=1)

        self.data_df.reset_index(drop=True, inplace=True)
        self.logger.info("Price fluctuations calculated.")
        return self.data_df

    def calculate_technical_indicators(self):
        """
        Compute RSI, MACD, and rate-of-change (ROC) for the 'Close' column.
        """
        if 'Close' not in self.data_df.columns:
            self.logger.warning("'Close' column not found. Skipping technical indicator calculations.")
            return self.data_df

        self.data_df['Close'] = pd.to_numeric(self.data_df['Close'], errors='coerce')

        # Prepare a dictionary for the new columns
        new_columns = {}

        # Compute RSI and MACD
        self.logger.info("Calculating RSI and MACD indicators...")
        new_columns['RSI'] = talib.RSI(self.data_df['Close'], timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(
            self.data_df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        new_columns['MACD'] = macd
        new_columns['MACD_Signal'] = macd_signal
        new_columns['MACD_Hist'] = macd_hist

        # Compute rate of change (ROC) for RSI and MACD_Signal
        for column in ['RSI', 'MACD_Signal']:
            if column in new_columns:
                self.logger.info(f"Calculating rate of change (ROC) for {column}...")
                new_columns[f"{column}_roc"] = pd.Series(new_columns[column]).diff().fillna(0)
            else:
                self.logger.warning(f"Column {column} not found. Skipping ROC calculation.")

        # Add all new columns to the DataFrame at once
        self.data_df = pd.concat([self.data_df, pd.DataFrame(new_columns, index=self.data_df.index)], axis=1)

        self.logger.info("Technical indicators calculated successfully.")
        return self.data_df

    def analyze_market(self, max_gather_minutes, step=5):
        """
        High-level function to calculate both price fluctuations and technical indicators.
        :param max_gather_minutes: Maximum gather time horizon in minutes.
        :param step: Step size for intervals in minutes (default is 5).
        :return: Updated DataFrame with all market analysis columns.
        """
        self.logger.info("Starting market analysis...")
        self.calculate_price_fluctuations(max_gather_minutes, step)
        self.calculate_technical_indicators()
        self.logger.info("Market analysis completed.")
        return self.data_df
