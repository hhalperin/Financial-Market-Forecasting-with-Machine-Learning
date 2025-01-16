import pandas as pd
import numpy as np
import talib
from utils.logger import get_logger


class MarketAnalyzer:
    """
    Handles market-related calculations, including:
    - Price fluctuations over specified time horizons.
    - Technical indicators like RSI, MACD, and rate of change (ROC).
    """

    def __init__(self, data_df):
        """
        :param data_df: A DataFrame containing stock price data with a 'Close' column.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.data_df = data_df.copy()

    def calculate_price_fluctuations(self, time_horizons):
        """
        Calculate price changes and percentage changes for given time horizons.
        :param time_horizons: List of dictionaries, each with:
                              - 'target_name': Name for the new columns.
                              - 'time_horizon': timedelta object specifying the horizon.
        :return: Updated DataFrame with fluctuation columns added.
        """
        if 'DateTime' not in self.data_df.columns or 'Close' not in self.data_df.columns:
            self.logger.error("Required columns 'DateTime' or 'Close' are missing.")
            return self.data_df

        self.data_df.set_index('DateTime', inplace=True, drop=False)
        self.data_df['Close'] = pd.to_numeric(self.data_df['Close'], errors='coerce')

        for config in time_horizons:
            base_name = config['target_name']
            minutes = int(config['time_horizon'].total_seconds() // 60)

            shifted_close = self.data_df['Close'].shift(-minutes)
            self.data_df[f"{base_name}_change"] = shifted_close - self.data_df['Close']
            self.data_df[f"{base_name}_percentage_change"] = (
                (shifted_close - self.data_df['Close']) / self.data_df['Close'] * 100
            )

        self.data_df.reset_index(drop=True, inplace=True)
        self.logger.info("Price fluctuations calculated for all specified time horizons.")
        return self.data_df

    def calculate_technical_indicators(self):
        """
        Compute RSI, MACD, and rate-of-change (ROC) for the 'Close' column.
        :return: Updated DataFrame with technical indicator columns added.
        """
        if 'Close' not in self.data_df.columns:
            self.logger.warning("'Close' column not found. Skipping technical indicator calculations.")
            return self.data_df

        self.data_df['Close'] = pd.to_numeric(self.data_df['Close'], errors='coerce')

        # Compute RSI and MACD
        self.logger.info("Calculating RSI and MACD indicators...")
        self.data_df['RSI'] = talib.RSI(self.data_df['Close'], timeperiod=14)
        self.data_df['MACD'], self.data_df['MACD_Signal'], self.data_df['MACD_Hist'] = talib.MACD(
            self.data_df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
        )

        # Compute rate of change (ROC) for RSI and MACD_Signal
        for column in ['RSI', 'MACD_Signal']:
            if column in self.data_df.columns:
                self.logger.info(f"Calculating rate of change (ROC) for {column}...")
                self.data_df[column] = self.data_df[column].fillna(0)
                self.data_df[f"{column}_roc"] = self.data_df[column].diff().fillna(0)
            else:
                self.logger.warning(f"Column {column} not found. Skipping ROC calculation.")

        self.logger.info("Technical indicators calculated successfully.")
        return self.data_df

    def analyze_market(self, time_horizons):
        """
        High-level function to calculate both price fluctuations and technical indicators.
        :param time_horizons: List of dictionaries specifying fluctuation time horizons.
        :return: Updated DataFrame with all market analysis columns.
        """
        self.logger.info("Starting market analysis...")
        self.calculate_price_fluctuations(time_horizons)
        self.calculate_technical_indicators()
        self.logger.info("Market analysis completed.")
        return self.data_df
