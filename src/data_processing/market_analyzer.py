"""
Market Analyzer Module

Computes market-related features including price fluctuations and technical indicators
(RSI, MACD, ROC) for stock price data.
"""

import pandas as pd
import numpy as np
import talib
from typing import Any
from src.utils.logger import get_logger

class MarketAnalyzer:
    """
    Computes market analysis features for stock price data.
    """
    def __init__(self, data_df: pd.DataFrame) -> None:
        """
        Initializes the MarketAnalyzer.

        :param data_df: DataFrame containing stock price data (requires 'DateTime' and 'Close' columns).
        """
        self.logger = get_logger(self.__class__.__name__)
        self.data_df: pd.DataFrame = data_df.copy()

    def calculate_price_fluctuations(self, max_gather_minutes: int, step: int = 5) -> pd.DataFrame:
        """
        Calculates price changes and percentage changes for various time intervals.

        :param max_gather_minutes: Maximum horizon in minutes.
        :param step: Step interval in minutes.
        :return: DataFrame with new fluctuation feature columns added.
        """
        if 'DateTime' not in self.data_df.columns or 'Close' not in self.data_df.columns:
            self.logger.error("Required columns 'DateTime' or 'Close' are missing.")
            return self.data_df

        self.data_df.set_index('DateTime', inplace=True, drop=False)
        self.data_df['Close'] = pd.to_numeric(self.data_df['Close'], errors='coerce')

        new_columns = {}
        for minutes in range(step, max_gather_minutes + 1, step):
            self.logger.info(f"Calculating price fluctuation for {minutes} minutes.")
            shifted_close = self.data_df['Close'].shift(-minutes)
            new_columns[f"{minutes}_minutes_change"] = shifted_close - self.data_df['Close']
            new_columns[f"{minutes}_minutes_percentage_change"] = ((shifted_close - self.data_df['Close']) / self.data_df['Close'] * 100)

        fluct_df = pd.DataFrame(new_columns, index=self.data_df.index)
        self.data_df = pd.concat([self.data_df, fluct_df], axis=1)
        self.data_df.reset_index(drop=True, inplace=True)
        self.logger.info("Price fluctuations calculated.")
        return self.data_df

    def calculate_technical_indicators(self) -> pd.DataFrame:
        """
        Computes RSI, MACD, and rate-of-change (ROC) indicators for the price data.

        :return: DataFrame with technical indicator columns added.
        """
        if 'Close' not in self.data_df.columns:
            self.logger.warning("'Close' column not found. Skipping technical indicator calculations.")
            return self.data_df

        self.data_df['Close'] = pd.to_numeric(self.data_df['Close'], errors='coerce')
        self.logger.info("Calculating RSI and MACD indicators...")
        rsi = talib.RSI(self.data_df['Close'], timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(self.data_df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

        # Compute ROC for RSI and MACD_Signal.
        rsi_roc = pd.Series(rsi).diff().fillna(0)
        macd_signal_roc = pd.Series(macd_signal).diff().fillna(0)

        technical_df = pd.DataFrame({
            'RSI': rsi,
            'MACD': macd,
            'MACD_Signal': macd_signal,
            'MACD_Hist': macd_hist,
            'RSI_roc': rsi_roc,
            'MACD_Signal_roc': macd_signal_roc
        }, index=self.data_df.index)

        self.data_df = pd.concat([self.data_df, technical_df], axis=1)
        self.logger.info("Technical indicators calculated successfully.")
        return self.data_df

    def analyze_market(self, max_gather_minutes: int, step: int = 5) -> pd.DataFrame:
        """
        Runs both price fluctuation and technical indicator calculations.

        :param max_gather_minutes: Maximum time horizon (in minutes) for fluctuation analysis.
        :param step: Interval step (in minutes).
        :return: Updated DataFrame with added market analysis features.
        """
        self.logger.info("Starting market analysis...")
        self.calculate_price_fluctuations(max_gather_minutes, step)
        self.calculate_technical_indicators()
        self.logger.info("Market analysis completed.")
        return self.data_df
