import pandas as pd
import logging
from utils.logger import get_logger

class DataPreprocessor:
    def __init__(self, price_df, news_df):
        self.price_df = price_df
        self.news_df = news_df
        self.logger = get_logger(self.__class__.__name__)

    def align_data(self):
        # Ensure DateTime columns are properly formatted
        self.price_df['DateTime'] = pd.to_datetime(self.price_df['DateTime'])
        self.news_df['time_published'] = pd.to_datetime(self.news_df['time_published'])

        # Merge on DateTime, perhaps with a tolerance for matching times
        merged_df = pd.merge_asof(
            self.price_df.sort_values('DateTime'),
            self.news_df.sort_values('time_published'),
            left_on='DateTime',
            right_on='time_published',
            direction='nearest',
            tolerance=pd.Timedelta('1min')
        )
        self.logger.info(f"Merged DataFrame shape: {merged_df.shape}")
        return merged_df

    def clean_data(self, df):
        # Implement any cleaning steps
        # For example, drop rows with missing values
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        self.logger.info(f"Cleaned DataFrame shape: {df.shape}")
        return df
