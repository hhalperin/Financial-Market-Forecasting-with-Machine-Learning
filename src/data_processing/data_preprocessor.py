import pandas as pd
from logger import get_logger

class DataPreprocessor:
    def __init__(self, price_df, news_df):
        self.price_df = price_df
        self.news_df = news_df
        self.logger = get_logger(self.__class__.__name__)

    def align_data(self):
        """
        Aligns and merges price and news data on DateTime, ensuring only historical news is used.
        """
        self.price_df['DateTime'] = pd.to_datetime(self.price_df['DateTime'])
        self.news_df['time_published'] = pd.to_datetime(self.news_df['time_published'])

        # Merge price data with news articles, ensuring news is always historical.
        merged_df = pd.merge_asof(
            self.price_df.sort_values('DateTime'),
            self.news_df.sort_values('time_published'),
            left_on='DateTime',
            right_on='time_published',
            direction='backward'  # Only include news articles that occurred before or at the price point
        )
        self.logger.info(f"Merged DataFrame shape: {merged_df.shape}")
        return merged_df

    def clean_data(self, df):
        """
        Cleans the merged data using advanced imputation techniques.
        """
        # Fill missing values for prices using a rolling average.
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].rolling(window=5, min_periods=1).mean())

        # Handle missing values in technical indicators and sentiment scores.
        # Forward-fill technical indicators, and for sentiment, fill with 0 or 1 as appropriate.
        for col in df.columns:
            if 'RSI' in col or 'MACD' in col:
                df[col] = df[col].fillna(method='ffill').fillna(0)
            elif 'positive' in col or 'negative' in col or 'neutral' in col:
                df[col] = df[col].fillna(1.0 if 'neutral' in col else 0.0)

        self.logger.info(f"Cleaned DataFrame shape: {df.shape}")
        return df
