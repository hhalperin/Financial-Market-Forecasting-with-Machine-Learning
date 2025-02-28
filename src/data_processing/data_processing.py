"""
Data Processing Module

Orchestrates the cleaning, merging, feature engineering, sentiment analysis, 
and embedding generation of aggregated data. Utilizes helper modules:
  - SentimentProcessor for sentiment analysis.
  - MarketAnalyzer for computing technical indicators.
  - DataEmbedder for generating text embeddings.
  - TimeHorizonManager for generating training horizon configurations.
"""

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
    def __init__(
        self,
        price_df: pd.DataFrame,
        news_df: pd.DataFrame,
        sentiment_model: str = "ProsusAI/finbert",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        """
        Initializes the DataProcessor with price and news data and configures helper processors.

        :param price_df: DataFrame with price data (must include 'DateTime').
        :param news_df: DataFrame with news data (must include 'time_published').
        :param sentiment_model: Model for sentiment analysis.
        :param embedding_model: Model for generating text embeddings.
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

        self.df: pd.DataFrame = pd.DataFrame()          # Final merged DataFrame.
        self.numeric_df: pd.DataFrame = pd.DataFrame()    # Numeric features for training.

    def process_news(self) -> pd.DataFrame:
        """
        Processes news data: converts timestamps, fills missing values,
        computes sentiment, and generates embeddings.

        :return: Processed news DataFrame.
        """
        if self.news_df.empty:
            self.logger.warning("News DataFrame is empty; skipping news processing.")
            return pd.DataFrame()
        
        news_df = self.news_df.copy()
        news_df["time_published"] = pd.to_datetime(news_df["time_published"], errors="coerce").dt.floor("min")
        for col in ["title", "summary", "url", "authors"]:
            if col in news_df.columns:
                news_df[col] = news_df[col].fillna("")
        news_df.sort_values("time_published", inplace=True)
        
        def classify_session(dt):
            if dt.hour < 9 or (dt.hour == 9 and dt.minute < 30):
                return 0  # Pre-market.
            elif (dt.hour == 9 and dt.minute >= 30) or (10 <= dt.hour < 16):
                return 1  # Market hours.
            else:
                return 2  # After-hours.
        news_df["release_session"] = news_df["time_published"].apply(classify_session)
        
        for col in ["title", "summary"]:
            if col in news_df.columns:
                texts = news_df[col].tolist()
                pos, neg, neu, labels = self.sentiment_processor.analyze_sentiment(texts)
                news_df[f"{col}_positive"] = pos
                news_df[f"{col}_negative"] = neg
                news_df[f"{col}_neutral"] = neu
                news_df[f"{col}_sentiment"] = labels
        
        news_df = self.embedder.embed_columns(news_df, ["title", "summary"])
        self.logger.info("Completed news data processing.")
        return news_df

    def process_price(self, max_gather_minutes: int) -> pd.DataFrame:
        """
        Processes price data: cleans, converts timestamps, adds cyclical features,
        and computes technical indicators using MarketAnalyzer.

        :param max_gather_minutes: Maximum window (in minutes) for price aggregation.
        :return: Processed price DataFrame.
        """
        if self.price_df.empty:
            self.logger.warning("Price DataFrame is empty; skipping price processing.")
            return pd.DataFrame()
        
        price_df = self.price_df.copy()
        price_df["DateTime"] = pd.to_datetime(price_df["DateTime"], errors="coerce")
        price_df.dropna(subset=["DateTime"], inplace=True)
        price_df.sort_values("DateTime", inplace=True)
        
        price_df["hour"] = price_df["DateTime"].dt.hour
        price_df["day_of_week"] = price_df["DateTime"].dt.dayofweek
        price_df["hour_sin"] = np.sin(2 * np.pi * price_df["hour"] / 24)
        price_df["hour_cos"] = np.cos(2 * np.pi * price_df["hour"] / 24)
        price_df["dow_sin"] = np.sin(2 * np.pi * price_df["day_of_week"] / 7)
        price_df["dow_cos"] = np.cos(2 * np.pi * price_df["day_of_week"] / 7)
        
        from .market_analyzer import MarketAnalyzer
        analyzer = MarketAnalyzer(price_df)
        price_df = analyzer.analyze_market(max_gather_minutes, step=settings.time_horizon_step)
        self.logger.info("Completed price data processing.")
        return price_df

    def merge_data(self, price_df: pd.DataFrame, news_df: pd.DataFrame, tolerance: str = "3h", direction: str = "backward") -> pd.DataFrame:
        """
        Merges price and news data using a temporal merge (merge_asof) and drops rows without news.

        :param price_df: Processed price DataFrame.
        :param news_df: Processed news DataFrame.
        :param tolerance: Maximum allowed time difference (e.g., "3h").
        :param direction: Merge direction ("backward", "forward", "nearest").
        :return: Merged DataFrame.
        """
        if price_df.empty or news_df.empty:
            self.logger.warning("One of the DataFrames is empty; cannot perform merge.")
            return pd.DataFrame()
        
        merged_df = pd.merge_asof(
            left=price_df.sort_values("DateTime"),
            right=news_df.sort_values("time_published"),
            left_on="DateTime",
            right_on="time_published",
            direction=direction,
            tolerance=pd.Timedelta(tolerance)
        )
        merged_df = merged_df.dropna(subset=["time_published"])
        self.logger.info(f"Merged data shape after filtering: {merged_df.shape}")
        return merged_df

    @profile_time(threshold=90.0)
    def process_pipeline(self, time_horizons: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Orchestrates the full data processing pipeline:
          - Processes news data.
          - Processes price data.
          - Merges the two DataFrames.
          - Computes expected sentiment.
        
        :param time_horizons: List of time horizon configurations.
        :return: Final processed DataFrame.
        """
        if not time_horizons:
            raise ValueError("No time horizons provided; please generate them before processing.")
        
        self.logger.info("Starting full data processing pipeline...")
        max_gather_minutes = max(int(cfg["time_horizon"].total_seconds() // 60) for cfg in time_horizons)
        
        news_processed = self.process_news()
        price_processed = self.process_price(max_gather_minutes)
        merged_df = self.merge_data(price_processed, news_processed, tolerance="3h", direction="backward")
        
        if merged_df.empty:
            self.logger.warning("Merged DataFrame is empty; no data to process further.")
            self.df = pd.DataFrame()
            self.numeric_df = pd.DataFrame()
            return self.df
        
        merged_df = self.sentiment_processor.compute_expected_sentiment(merged_df)
        self.df = merged_df
        self.numeric_df = merged_df.select_dtypes(include=["number"])
        self.logger.info(f"Final processed DataFrame shape: {self.df.shape}")
        self.logger.info(f"Numeric DataFrame for training shape: {self.numeric_df.shape}")
        return self.df
