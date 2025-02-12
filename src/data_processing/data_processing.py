"""
Data Processing Pipeline Module

The DataProcessor class orchestrates the end-to-end processing of aggregated data.
It performs the following steps:
  - Clean and sort price data.
  - Preprocess news data.
  - Analyze market features (price fluctuations and technical indicators).
  - Merge price and news data using as-of joins.
  - Drop incomplete news records.
  - Perform sentiment analysis on news texts.
  - Generate embeddings for text fields (replacing original text with a single embedding column).

Common models (sentiment and embedding) are configurable via the Settings.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import talib
from typing import List, Dict, Any
from src.utils.logger import get_logger
from src.performance_monitor import profile_time
from .sentiment_processor import SentimentProcessor
from .market_analyzer import MarketAnalyzer
from .data_embedder import DataEmbedder

pd.set_option('future.no_silent_downcasting', True)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

class DataProcessor:
    """
    Processes stock price and news data through a series of steps to create both a raw processed
    DataFrame and a numeric DataFrame for model training.
    """
    def __init__(self, price_df: pd.DataFrame, news_df: pd.DataFrame, 
                 sentiment_model: str = "ProsusAI/finbert", 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """
        Initializes the DataProcessor.

        :param price_df: DataFrame containing stock price data.
        :param news_df: DataFrame containing news articles.
        :param sentiment_model: Model name for sentiment analysis.
        :param embedding_model: Model name for embedding generation.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.price_df = price_df.copy() if price_df is not None else pd.DataFrame()
        self.news_df = news_df.copy() if news_df is not None else pd.DataFrame()
        self.sentiment_processor = SentimentProcessor(model_name=sentiment_model)
        self.embedder = DataEmbedder(model_name=embedding_model)
        # The merged raw DataFrame.
        self.df: pd.DataFrame = pd.DataFrame()
        # Numeric DataFrame for training, with text columns replaced by embedding vectors.
        self.numeric_df: pd.DataFrame = pd.DataFrame()

    def clean_price_data(self) -> None:
        """Cleans and sorts the price data by 'DateTime'."""
        self.price_df['DateTime'] = pd.to_datetime(self.price_df['DateTime'], errors='coerce')
        self.price_df.dropna(subset=['DateTime'], inplace=True)
        self.price_df.sort_values('DateTime', inplace=True)
        self.logger.info(f"Cleaned price_df. Shape: {self.price_df.shape}")

    def preprocess_news(self) -> None:
        """Converts 'time_published' to datetime, drops missing values, and sorts the news data."""
        self.news_df['time_published'] = pd.to_datetime(self.news_df['time_published'], errors='coerce').dt.floor('min')
        self.news_df.dropna(subset=['time_published'], inplace=True)
        self.news_df.sort_values('time_published', inplace=True)
        self.logger.info(f"Preprocessed news_df. Shape: {self.news_df.shape}")

    def perform_market_analysis(self, max_gather_minutes: int, step: int = 5) -> None:
        """
        Enhances the price data with market analysis features using MarketAnalyzer.
        
        :param max_gather_minutes: Maximum time horizon (in minutes) for price fluctuation calculations.
        :param step: Time step (in minutes) for generating fluctuation features.
        """
        analyzer = MarketAnalyzer(self.price_df)
        self.price_df = analyzer.analyze_market(max_gather_minutes, step)
        self.logger.info(f"Market analysis completed. Updated price_df shape: {self.price_df.shape}")

    def merge_data_asof(self, tolerance: str = "5min", direction: str = "backward") -> None:
        """
        Merges price and news data using an as-of join based on time proximity.

        :param tolerance: Maximum time difference allowed between price and news data.
        :param direction: Direction for the as-of join.
        """
        if self.price_df.empty or self.news_df.empty:
            self.logger.warning("Either price_df or news_df is empty; cannot perform merge.")
            self.df = pd.DataFrame()
            return

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
        Drops rows from the merged DataFrame that are missing essential news fields ('title' and 'summary').
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
        Performs sentiment analysis on the 'title' and 'summary' columns and computes an expected sentiment.
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
        Generates embeddings for specified text columns, replacing them with a single embedding column.
        
        :param columns_to_embed: List of column names to embed. Defaults to ['title', 'summary'].
        """
        if self.df.empty:
            self.logger.warning("No data available for embedding generation.")
            return

        if not columns_to_embed:
            columns_to_embed = ['title', 'summary', 'authors']

        # Replace each text column with a new column containing its embedding vector.
        self.numeric_df = self.df.copy()
        self.numeric_df = self.embedder.embed_columns(self.numeric_df, columns_to_embed)
        self.logger.info(f"Numeric DataFrame generated for training. Shape: {self.numeric_df.shape}")

    @profile_time(threshold=1.0)
    def process_pipeline(self, time_horizons: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Executes the full processing pipeline in sequence:
          1. Clean price data.
          2. Perform market analysis.
          3. Preprocess news data.
          4. Merge price and news data.
          5. Drop incomplete news rows.
          6. Process sentiment.
          7. Generate embeddings.

        :param time_horizons: List of time horizon configurations.
        :return: The final processed DataFrame.
        """
        self.logger.info("Starting full data processing pipeline...")
        if not time_horizons:
            raise ValueError("No time horizons provided; please generate them before processing.")

        max_gather_minutes = max(int(cfg['time_horizon'].total_seconds() // 60) for cfg in time_horizons)

        self.clean_price_data()
        self.perform_market_analysis(max_gather_minutes, step=5)
        self.preprocess_news()
        self.merge_data_asof(tolerance="5min", direction="backward")
        self.drop_incomplete_news()
        self.process_sentiment()
        self.generate_embeddings(columns_to_embed=['title', 'summary'])

        self.logger.info(f"Final processed DataFrame (raw) shape: {self.df.shape}")
        self.logger.info(f"Final numeric DataFrame (for training) shape: {self.numeric_df.shape}")
        return self.df
