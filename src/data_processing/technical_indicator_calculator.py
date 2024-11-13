import pandas as pd
import logging
from utils.logger import get_logger

class TechnicalIndicatorCalculator:
    """
    Calculates additional technical indicators or derivatives.
    """

    def __init__(self, data_df):
        self.data_df = data_df.copy()
        self.logger = get_logger(self.__class__.__name__)

    def calculate_rate_of_change(self, columns):
        for column in columns:
            if column in self.data_df.columns:
                self.logger.info(f"Calculating rate of change for {column}")
                
                # Convert column to numeric
                self.data_df[column] = pd.to_numeric(self.data_df[column], errors='coerce')
                
                # Forward fill missing values
                self.data_df[column] = self.data_df[column].fillna(method='ffill')
                
                # Drop remaining NaN values
                self.data_df = self.data_df[self.data_df[column].notna()]
                
                roc_column = f'{column}_roc'
                self.data_df[roc_column] = self.data_df[column].diff()
                
                # Fill NaN values in roc_column with zero
                self.data_df[roc_column] = self.data_df[roc_column].fillna(0)
            else:
                self.logger.warning(f"Column {column} not found in DataFrame. Skipping rate of change calculation.")
        return self.data_df
