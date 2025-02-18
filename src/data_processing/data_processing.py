"""
Data Processor Module

The DataProcessor class orchestrates the end-to-end processing of aggregated data.
It performs the following steps:
  - Clean and sort price data.
  - Preprocess news data.
  - Analyze market features (price fluctuations and technical indicators).
  - Merge price and news data using as-of joins.
  - Drop incomplete news records.
  - Perform sentiment analysis on news texts.
  - Generate embeddings for text fields (with optional composite embedding).
  - Add time-based features from price and news data.
  - Classify news release times into market sessions (pre-market, market-hours, after-market).
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Dict, Any

from src.utils.logger import get_logger
from src.utils.performance_monitor import profile_time  # Assumes an external decorator for profiling time.
from .sentiment_processor import SentimentProcessor
from .market_analyzer import MarketAnalyzer
from .data_embedder import DataEmbedder
from src.config import settings

class DataProcessor:
    """
    Orchestrates the complete data processing pipeline for merging price and news data,
    performing sentiment analysis, generating embeddings, and adding time-based features.
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        news_df: pd.DataFrame,
        sentiment_model: str = "ProsusAI/finbert",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        """
        Initializes the DataProcessor with copies of price and news data, and sets up helper processors.

        :param price_df: Raw price data DataFrame.
        :param news_df: Raw news data DataFrame.
        :param sentiment_model: Model identifier for sentiment analysis.
        :param embedding_model: Model identifier for generating text embeddings.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.price_df = price_df.copy() if price_df is not None else pd.DataFrame()
        self.news_df = news_df.copy() if news_df is not None else pd.DataFrame()
        # Initialize SentimentProcessor with recency weighting parameters from config.
        self.sentiment_processor = SentimentProcessor(
            model_name=sentiment_model,
            use_recency_weighting=settings.sentiment_use_recency_weighting,
            recency_decay=settings.sentiment_recency_decay
        )
        # Initialize DataEmbedder with new composite embedding parameters from config.
        self.embedder = DataEmbedder(
            model_name=embedding_model,
            n_components=settings.embedding_n_components,
            batch_size=settings.embedding_batch_size,
            use_pca=settings.embedding_use_pca,
            combine_fields=settings.embedding_combine_fields,
            fields_to_combine=settings.embedding_fields_to_combine,
            combine_template=settings.embedding_combine_template
        )
        self.df: pd.DataFrame = pd.DataFrame()         # Final merged DataFrame.
        self.numeric_df: pd.DataFrame = pd.DataFrame()   # DataFrame for numeric (training) data.

    def clean_price_data(self) -> None:
        """
        Cleans and sorts the price data.
        - Converts 'DateTime' column to datetime objects.
        - Drops rows with invalid datetime values.
        - Sorts the DataFrame by 'DateTime'.
        """
        self.price_df['DateTime'] = pd.to_datetime(self.price_df['DateTime'], errors='coerce')
        self.price_df.dropna(subset=['DateTime'], inplace=True)
        self.price_df.sort_values('DateTime', inplace=True)
        self.logger.info(f"Cleaned price_df. Shape: {self.price_df.shape}")

    def preprocess_news(self) -> None:
        """
        Preprocesses the news data.
        - Converts 'time_published' column to datetime objects (floored to minutes).
        - Drops rows with invalid datetime values.
        - Sorts the DataFrame by 'time_published'.
        """
        self.news_df['time_published'] = pd.to_datetime(self.news_df['time_published'], errors='coerce').dt.floor('min')
        self.news_df.dropna(subset=['time_published'], inplace=True)
        self.news_df.sort_values('time_published', inplace=True)
        self.logger.info(f"Preprocessed news_df. Shape: {self.news_df.shape}")

    def perform_market_analysis(self, max_gather_minutes: int, step: int = 5) -> None:
        """
        Analyzes market features using the MarketAnalyzer.
        Updates the price DataFrame with computed technical indicators and price fluctuations.

        :param max_gather_minutes: Maximum minutes for gathering market data.
        :param step: Step size in minutes for the analysis.
        """
        analyzer = MarketAnalyzer(self.price_df)
        self.price_df = analyzer.analyze_market(max_gather_minutes, step)
        self.logger.info(f"Market analysis completed. Updated price_df shape: {self.price_df.shape}")

    def merge_data_asof(self, tolerance: str = "5min", direction: str = "backward") -> None:
        """
        Performs an as-of merge between price and news data.
        Aligns news items with the nearest price record within the specified tolerance.

        :param tolerance: Maximum allowed time difference (e.g., '5min').
        :param direction: Merge direction ('backward', 'forward', or 'nearest').
        """
        if self.price_df.empty or self.news_df.empty:
            self.logger.warning("Either price_df or news_df is empty; cannot perform merge.")
            self.df = pd.DataFrame()
            return

        # Ensure datetime columns are in proper datetime format.
        price = self.price_df.copy()
        news = self.news_df.copy()
        price['DateTime'] = pd.to_datetime(price['DateTime'])
        news['time_published'] = pd.to_datetime(news['time_published'])
        price.sort_values('DateTime', inplace=True)
        news.sort_values('time_published', inplace=True)

        merged = pd.merge_asof(
            left=price,
            right=news,
            left_on='DateTime',
            right_on='time_published',
            direction=direction,
            tolerance=pd.Timedelta(tolerance)
        )
        self.df = merged
        self.logger.info(f"ASOF-merged data shape: {self.df.shape}")

    @profile_time(threshold=1.0)
    def drop_incomplete_news(self) -> None:
        """
        Drops rows from the merged DataFrame that are missing critical news fields.
        Specifically, drops rows missing values in 'title' or 'summary'.
        """
        if self.df.empty:
            self.logger.warning("No merged data available for dropping incomplete news.")
            return

        missing_cols = [c for c in ['title', 'summary'] if c not in self.df.columns]
        if missing_cols:
            self.logger.warning(f"Columns missing for dropna: {missing_cols}. Skipping drop.")
            return

        initial_count = len(self.df)
        self.df.dropna(subset=['title', 'summary'], how='any', inplace=True)
        final_count = len(self.df)
        self.logger.info(f"Dropped {initial_count - final_count} rows with missing 'title' or 'summary'.")

    @profile_time(threshold=1.0)
    def process_sentiment(self) -> None:
        """
        Applies sentiment analysis on specified text columns (e.g., 'title', 'summary').
        Adds new columns for positive, negative, and neutral sentiment scores, as well as a sentiment label.
        Computes an expected sentiment value using recency weighting if enabled.
        """
        if self.df.empty:
            self.logger.warning("No data available for sentiment analysis.")
            return

        for col in ['title', 'summary']:
            if col in self.df.columns:
                texts = self.df[col].fillna('').tolist()
                pos, neg, neu, labels = self.sentiment_processor.analyze_sentiment(texts)
                self.df[f"{col}_positive"] = pos
                self.df[f"{col}_negative"] = neg
                self.df[f"{col}_neutral"] = neu
                self.df[f"{col}_sentiment"] = labels

        self.df = self.sentiment_processor.compute_expected_sentiment(self.df)
        self.logger.info("Sentiment analysis completed.")

    @profile_time(threshold=1.0)
    def generate_embeddings(self, columns_to_embed: List[str] = None) -> None:
        """
        Generates numeric embeddings for the specified text columns.
        If composite embedding is enabled via config, combines specified fields into a single text.
        The resulting numeric DataFrame is intended for training purposes.

        :param columns_to_embed: List of column names to generate embeddings for.
                                 Defaults to ['title', 'summary'] if not provided.
        """
        if self.df.empty:
            self.logger.warning("No data available for embedding generation.")
            return

        if not columns_to_embed:
            columns_to_embed = ['title', 'summary']
        # Generate embeddings (composite or individual based on configuration)
        self.numeric_df = self.df.copy()
        self.numeric_df = self.embedder.embed_columns(self.numeric_df, columns_to_embed)
        self.logger.info(f"Numeric DataFrame generated for training. Shape: {self.numeric_df.shape}")

    def add_price_time_features(self) -> None:
        """
        Extracts cyclical time features from the price DataFrame's 'DateTime' column.
        Adds hour-based and day-of-week based sine/cosine features.
        """
        if 'DateTime' not in self.price_df.columns:
            self.logger.warning("No 'DateTime' column in price_df to extract time features.")
            return
        df = self.price_df.copy()
        df['hour'] = df['DateTime'].dt.hour
        df['day_of_week'] = df['DateTime'].dt.dayofweek
        # Convert hour into cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        # Day-of-week cyclical features
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        self.price_df = df
        self.logger.info("Price time features added.")

    def add_news_release_session(self) -> None:
        """
        Classifies news release times into market sessions:
          0 - Pre-market (before 9:30 AM),
          1 - Market hours (9:30 AM to 16:00),
          2 - After-market (after 16:00).
        Adds a new 'release_session' column to the news DataFrame.
        """
        if 'time_published' not in self.news_df.columns:
            self.logger.warning("No 'time_published' column in news_df to classify release sessions.")
            return

        def classify_session(dt):
            hour = dt.hour
            minute = dt.minute
            if hour < 9 or (hour == 9 and minute < 30):
                return 0
            elif (hour == 9 and minute >= 30) or (hour >= 10 and hour < 16):
                return 1
            else:
                return 2

        self.news_df['release_session'] = self.news_df['time_published'].apply(classify_session)
        self.logger.info("News release sessions classified (0: pre-market, 1: market, 2: after-market).")

    def process_pipeline(self, time_horizons: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Executes the complete data processing pipeline.
        Sequentially calls methods to clean, merge, analyze, and augment data.

        :param time_horizons: List of time horizon configurations used for market analysis.
        :return: The final merged and processed DataFrame.
        :raises ValueError: If no time horizons are provided.
        """
        self.logger.info("Starting full data processing pipeline...")
        if not time_horizons:
            raise ValueError("No time horizons provided; please generate them before processing.")

        max_gather_minutes = max(int(cfg['time_horizon'].total_seconds() // 60) for cfg in time_horizons)
        self.clean_price_data()
        self.add_price_time_features()
        self.perform_market_analysis(max_gather_minutes, step=5)
        self.preprocess_news()
        self.add_news_release_session()
        self.merge_data_asof(tolerance="5min", direction="backward")
        self.drop_incomplete_news()
        self.process_sentiment()
        self.generate_embeddings(columns_to_embed=['title', 'summary'])
        self.logger.info(f"Final processed DataFrame (raw) shape: {self.df.shape}")
        self.logger.info(f"Final numeric DataFrame (for training) shape: {self.numeric_df.shape}")
        return self.df
