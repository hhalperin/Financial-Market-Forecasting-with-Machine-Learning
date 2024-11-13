import pandas as pd
import logging
from utils.logger import get_logger

class PriceFluctuationCalculator:
    """
    Calculates price fluctuations for specified time periods.
    """

    def __init__(self, data_df, time_periods):
        self.data_df = data_df.copy()
        self.time_periods = time_periods
        self.logger = get_logger(self.__class__.__name__)

    def calculate_fluctuations(self):
        """
        Calculates price changes and percentage changes for each time period.
        """
        self.data_df.set_index('DateTime', inplace=True)
        self.data_df['Close'] = pd.to_numeric(self.data_df['Close'], errors='coerce')

        for label, minutes in self.time_periods.items():
            shifted_close = self.data_df['Close'].shift(-minutes)
            self.data_df[f'{label}_change'] = shifted_close - self.data_df['Close']
            self.data_df[f'{label}_percentage_change'] = ((shifted_close - self.data_df['Close']) / self.data_df['Close']) * 100

        self.data_df.reset_index(inplace=True)
        self.logger.info("Calculated price fluctuations.")
        return self.data_df
