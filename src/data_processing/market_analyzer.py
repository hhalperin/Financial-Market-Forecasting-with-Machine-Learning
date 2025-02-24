# src/market_analyzer.py

import pandas as pd
import numpy as np
from tqdm import tqdm
import talib
from typing import Any
from src.utils.logger import get_logger
import gc

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
        # Work on a copy to preserve the original data.
        self.data_df: pd.DataFrame = data_df.copy()

    def calculate_price_fluctuations(self, max_gather_minutes: int, step: int = 5, 
                                     use_chunking: bool = False, chunk_size: int = 10000) -> pd.DataFrame:
        """
        Calculates percentage price changes for various time intervals.
        
        If use_chunking is False, uses a fully vectorized approach (faster, but may require more memory).
        If use_chunking is True, processes the DataFrame in chunks to reduce peak memory usage,
        but with additional overhead.
        
        :param max_gather_minutes: Maximum horizon in minutes.
        :param step: Step interval in minutes.
        :param use_chunking: Whether to process the DataFrame in chunks.
        :param chunk_size: Number of rows to process per chunk (only used if use_chunking=True).
        :return: DataFrame with new fluctuation feature columns added.
        """
        if 'DateTime' not in self.data_df.columns or 'Close' not in self.data_df.columns:
            self.logger.error("Required columns 'DateTime' or 'Close' are missing.")
            return self.data_df

        # Set DateTime as index for easier shifting; ensure 'Close' is numeric.
        self.data_df.set_index('DateTime', inplace=True, drop=False)
        self.data_df['Close'] = pd.to_numeric(self.data_df['Close'], errors='coerce')

        if not use_chunking:
            # Vectorized approach (fast)
            new_columns = {}
            for minutes in tqdm(range(step, max_gather_minutes + 1, step), desc="Calculating Price Fluctuations"):
                shifted_close = self.data_df['Close'].shift(-minutes)
                new_columns[f"{minutes}_minutes_percentage_change"] = ((shifted_close - self.data_df['Close']) / self.data_df['Close'] * 100)
            fluct_df = pd.DataFrame(new_columns, index=self.data_df.index)
            self.data_df = pd.concat([self.data_df, fluct_df], axis=1)
        else:
            # Chunked processing (lower peak memory usage, but slower)
            total_rows = len(self.data_df)
            for minutes in tqdm(range(step, max_gather_minutes + 1, step), desc="Calculating Price Fluctuations"):
                col_name = f"{minutes}_minutes_percentage_change"
                # Initialize the new column with NaN values.
                self.data_df[col_name] = np.nan
                # Compute the shifted Series once for this interval.
                shifted_series = self.data_df['Close'].shift(-minutes)
                for start in range(0, total_rows, chunk_size):
                    end = min(start + chunk_size, total_rows)
                    idx = self.data_df.index[start:end]
                    self.data_df.loc[idx, col_name] = (
                        (shifted_series.loc[idx] - self.data_df.loc[idx, 'Close']) / self.data_df.loc[idx, 'Close'] * 100
                    )
                    del idx
                    gc.collect()
                del shifted_series
                gc.collect()

        # Reset index to restore original order.
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

        # Compute ROC for RSI and MACD Signal.
        rsi_roc = pd.Series(rsi).diff().fillna(0)
        macd_signal_roc = pd.Series(macd_signal).diff().fillna(0)

        # In-place assignment of technical indicator columns.
        self.data_df['RSI'] = rsi
        self.data_df['MACD'] = macd
        self.data_df['MACD_Signal'] = macd_signal
        self.data_df['MACD_Hist'] = macd_hist
        self.data_df['RSI_roc'] = rsi_roc
        self.data_df['MACD_Signal_roc'] = macd_signal_roc

        del rsi, macd, macd_signal, macd_hist, rsi_roc, macd_signal_roc
        gc.collect()

        self.logger.info("Technical indicators calculated successfully.")
        return self.data_df

    def analyze_market(self, max_gather_minutes: int, step: int = 5, 
                       use_chunking: bool = False, chunk_size: int = 10000) -> pd.DataFrame:
        """
        Runs both price fluctuation and technical indicator calculations.
        
        :param max_gather_minutes: Maximum time horizon (in minutes) for fluctuation analysis.
        :param step: Interval step (in minutes).
        :param use_chunking: Whether to process in chunks (default is False, for speed).
        :param chunk_size: Number of rows per chunk if chunking is used.
        :return: Updated DataFrame with added market analysis features.
        """
        self.logger.info("Starting market analysis...")
        self.calculate_price_fluctuations(max_gather_minutes, step, use_chunking, chunk_size)
        self.calculate_technical_indicators()
        self.logger.info("Market analysis completed.")
        return self.data_df
