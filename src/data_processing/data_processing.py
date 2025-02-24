# src/data_processing/data_processing.py
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Dict, Any

from src.utils.logger import get_logger
from src.utils.performance_monitor import profile_time 
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
        """
        self.logger = get_logger(self.__class__.__name__)
        self.price_df = price_df.copy() if price_df is not None else pd.DataFrame()
        self.news_df = news_df.copy() if news_df is not None else pd.DataFrame()
        self.sentiment_processor = SentimentProcessor(
            model_name=sentiment_model,
            use_recency_weighting=settings.sentiment_use_recency_weighting,
            recency_decay=settings.sentiment_recency_decay
        )
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
        """
        self.price_df['DateTime'] = pd.to_datetime(self.price_df['DateTime'], errors='coerce')
        self.price_df.dropna(subset=['DateTime'], inplace=True)
        self.price_df.sort_values('DateTime', inplace=True)
        self.logger.info(f"Cleaned price_df. Shape: {self.price_df.shape}")

    def preprocess_news(self) -> None:
        """
        Preprocesses the news data.
        """
        self.news_df['time_published'] = pd.to_datetime(self.news_df['time_published'], errors='coerce').dt.floor('min')
        # Fill missing news fields with empty strings to ensure they can be merged.
        for col in ['title', 'summary', 'url', 'authors']:
            if col in self.news_df.columns:
                self.news_df[col] = self.news_df[col].fillna("")
        self.news_df.sort_values('time_published', inplace=True)
        self.logger.info(f"Preprocessed news_df. Shape: {self.news_df.shape}")

    def perform_market_analysis(self, max_gather_minutes: int, step: int = 5) -> None:
        """
        Analyzes market features using the MarketAnalyzer.
        """
        analyzer = MarketAnalyzer(self.price_df)
        self.price_df = analyzer.analyze_market(max_gather_minutes, step)
        self.logger.info(f"Market analysis completed. Updated price_df shape: {self.price_df.shape}")

    def merge_data_asof(self, tolerance: str = "3h", direction: str = "backward") -> None:
        """
        Performs an as-of merge between price and news data using a longer tolerance.
        This ensures that even if news are gathered early, they will still be merged with later price data.
        """
        if self.price_df.empty:
            self.logger.warning("Price data is empty; cannot perform merge.")
            self.df = pd.DataFrame()
            return

        price = self.price_df.copy()
        news = self.news_df.copy()
        price['DateTime'] = pd.to_datetime(price['DateTime'])
        news['time_published'] = pd.to_datetime(news['time_published'])
        price.sort_values('DateTime', inplace=True)
        news.sort_values('time_published', inplace=True)
        self.df = pd.merge_asof(
            left=price,
            right=news,
            left_on='DateTime',
            right_on='time_published',
            direction=direction,
            tolerance=pd.Timedelta(tolerance)
        )
        # After merging, fill missing news fields with empty strings
        for col in ['title', 'summary', 'url', 'authors']:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("")
        self.logger.info(f"ASOF-merged data shape: {self.df.shape}")
        # Log how many rows have non-empty news titles
        non_empty_news = self.df[self.df['title'] != ""].shape[0]
        self.logger.info(f"Rows with news data after merge: {non_empty_news}")

    @profile_time(threshold=1.0)
    def drop_incomplete_news(self) -> None:
        """
        Instead of dropping rows with missing news, this method is skipped
        because missing news values are already filled.
        """
        self.logger.info("Skipping drop_incomplete_news since missing news values are filled.")

    @profile_time(threshold=1.0)
    def process_sentiment(self) -> None:
        """
        Applies sentiment analysis on text columns and computes expected sentiment.
        """
        if self.df.empty:
            self.logger.warning("No data available for sentiment analysis.")
            return

        for col in ['title', 'summary']:
            if col in self.df.columns:
                texts = self.df[col].tolist()
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
        Generates numeric embeddings for specified text columns.
        """
        if self.df.empty:
            self.logger.warning("No data available for embedding generation.")
            return

        if not columns_to_embed:
            columns_to_embed = ['title', 'summary']
        self.numeric_df = self.df.copy()
        self.numeric_df = self.embedder.embed_columns(self.numeric_df, columns_to_embed)
        self.logger.info(f"Numeric DataFrame generated for training. Shape: {self.numeric_df.shape}")

    def add_price_time_features(self) -> None:
        """
        Adds cyclical time features to the price DataFrame.
        """
        if 'DateTime' not in self.price_df.columns:
            self.logger.warning("No 'DateTime' column in price_df to extract time features.")
            return
        df = self.price_df.copy()
        df['hour'] = df['DateTime'].dt.hour
        df['day_of_week'] = df['DateTime'].dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        self.price_df = df
        self.logger.info("Price time features added.")

    def add_news_release_session(self) -> None:
        """
        Classifies news release times into market sessions.
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
        self.merge_data_asof(tolerance="3h", direction="backward")
        self.drop_incomplete_news()  # Now a no-op
        self.process_sentiment()
        self.generate_embeddings(columns_to_embed=['title', 'summary'])
        self.logger.info(f"Final processed DataFrame (raw) shape: {self.df.shape}")
        self.logger.info(f"Final numeric DataFrame (for training) shape: {self.numeric_df.shape}")
        return self.df
