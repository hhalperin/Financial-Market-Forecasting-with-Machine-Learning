import pandas as pd
from logger import get_logger

class TechnicalIndicatorCalculator:
    """
    Calculates additional technical indicators or derivatives.
    """

    def __init__(self, data_df):
        self.data_df = data_df.copy()
        self.logger = get_logger(self.__class__.__name__)

    def calculate_rate_of_change(self, columns):
        """
        Calculates rate of change for specified columns.
        """
        for column in columns:
            if column in self.data_df.columns:
                self.logger.info(f"Calculating rate of change for {column}")
                
                # Convert column to numeric and fill NaNs
                self.data_df[column] = pd.to_numeric(self.data_df[column], errors='coerce').fillna(method='ffill')

                roc_column = f'{column}_roc'
                self.data_df[roc_column] = self.data_df[column].diff().fillna(0)

            else:
                self.logger.warning(f"Column {column} not found in DataFrame. Skipping rate of change calculation.")
        
        self.logger.info("Completed calculation of technical indicators.")
        return self.data_df
