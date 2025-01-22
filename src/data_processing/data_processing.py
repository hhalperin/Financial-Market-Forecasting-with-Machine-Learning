import pandas as pd
import numpy as np
import talib
from src.utils.logger import get_logger
from .sentiment_processor import SentimentProcessor
from .market_analyzer import MarketAnalyzer
from .data_embedder import DataEmbedder

pd.set_option('future.no_silent_downcasting', True)

class DataProcessor:
    """
    Handles the end-to-end processing pipeline for merging, analyzing, and preparing data for ML.
    """

    def __init__(self, price_df, news_df, 
                 sentiment_model="ProsusAI/finbert", 
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """
        :param price_df: DataFrame with stock price data.
        :param news_df: DataFrame with news articles.
        :param sentiment_model: Hugging Face model for sentiment analysis.
        :param embedding_model: Hugging Face model for embedding generation.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.price_df = price_df.copy() if price_df is not None else pd.DataFrame()
        self.news_df = news_df.copy() if news_df is not None else pd.DataFrame()
        self.sentiment_processor = SentimentProcessor(model_name=sentiment_model)
        self.embedder = DataEmbedder(model_name=embedding_model)
        self.df = None  
        self.embeddings = None  

    def clean_price_data(self):
        """
        Cleans and prepares the price DataFrame:
        - Converts 'DateTime' to datetime and ensures proper sorting.
        """
        self.price_df['DateTime'] = pd.to_datetime(self.price_df['DateTime'], errors='coerce')
        self.price_df.dropna(subset=['DateTime'], inplace=True)
        self.price_df.sort_values('DateTime', inplace=True)
        self.logger.info(f"Cleaned price_df. Shape: {self.price_df.shape}")

    def preprocess_news(self):
        """
        Prepares the news DataFrame:
        - Converts 'time_published' to datetime (minute precision).
        - Removes rows where time is invalid.
        - Sorts by 'time_published'.
        """
        self.news_df['time_published'] = pd.to_datetime(self.news_df['time_published'], errors='coerce').dt.floor('min')
        self.news_df.dropna(subset=['time_published'], inplace=True)
        self.news_df.sort_values('time_published', inplace=True)
        self.logger.info(f"Preprocessed news_df. Shape: {self.news_df.shape}")

    def perform_market_analysis(self, max_gather_minutes, step=5):
        """
        Adds technical indicators and price fluctuations to the price DataFrame.
        :param max_gather_minutes: Maximum gather time horizon in minutes.
        :param step: Step size for intervals in minutes (default is 5).
        """
        analyzer = MarketAnalyzer(self.price_df)
        self.price_df = analyzer.analyze_market(max_gather_minutes, step)
        self.logger.info(f"Market analysis completed. Shape: {self.price_df.shape}")

    def merge_data_asof(self, tolerance="5min", direction="backward"):
        """
        Uses pandas.merge_asof to align each price row with the nearest news timestamp
        (within a specified tolerance). This avoids the 'mostly NaN' issue that arises
        from an exact-time outer join.

        :param tolerance: e.g. '5min' or '1H'. The max gap allowed to match news & price.
        :param direction: 'backward' means a price bar at 09:31 is matched to news at 09:31
                          or earlier (within tolerance). 
                          'forward' does the opposite.
        """
        if self.price_df.empty and self.news_df.empty:
            self.logger.warning("No price or news data to merge.")
            self.df = pd.DataFrame()
            return

        # Sort and rename columns for merge_asof
        price = self.price_df.copy()
        news = self.news_df.copy()

        price['DateTime'] = pd.to_datetime(price['DateTime'])
        news['time_published'] = pd.to_datetime(news['time_published'])

        price.sort_values('DateTime', inplace=True)
        news.sort_values('time_published', inplace=True)

        # Merge asof: left=price, right=news
        merged = pd.merge_asof(
            left=price,
            right=news,
            left_on='DateTime',
            right_on='time_published',
            direction=direction,
            tolerance=pd.Timedelta(tolerance)
        )

        # Now 'merged' has the same rows as price_df, each row matched to at most
        # one news article. If multiple price bars fall within tolerance of 
        # the same news timestamp, that news is repeated for each bar. 
        # That's normal for "price as left" merges.

        # Optionally rename 'time_published' back or keep as is
        # (We keep it here)
        self.df = merged
        self.logger.info(f"ASOF-merged data shape: {self.df.shape}")

    def drop_incomplete_news(self):
        """
        Remove rows from self.df that have no 'title' or 'summary' (if those columns exist).
        This ensures only "valid" articles remain.
        """
        if self.df is None or self.df.empty:
            self.logger.warning("No DataFrame in 'self.df' to drop incomplete news from.")
            return

        missing_cols = [c for c in ['title','summary'] if c not in self.df.columns]
        if missing_cols:
            self.logger.warning(f"These columns don't exist for dropping: {missing_cols}. Skipping drop.")
            return

        initial_count = len(self.df)
        self.df.dropna(subset=['title','summary'], how='any', inplace=True)
        final_count = len(self.df)
        self.logger.info(f"Dropped {initial_count - final_count} rows missing 'title' or 'summary'.")

    def process_sentiment(self):
        """
        Performs sentiment analysis on titles and summaries in the DataFrame.
        """
        if self.df is None or self.df.empty:
            self.logger.warning("No DataFrame available for sentiment analysis.")
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

    def generate_embeddings(self, columns_to_embed=None):
        """
        Generates embeddings for specified columns in the DataFrame.
        :param columns_to_embed: List of column names to embed.
        """
        if self.df is None or self.df.empty:
            self.logger.warning("No DataFrame available to generate embeddings.")
            return

        if not columns_to_embed:
            columns_to_embed = ['title', 'summary']

        # Only embed if those columns exist
        col_exists = [c for c in columns_to_embed if c in self.df.columns]
        if not col_exists:
            self.logger.warning("No valid columns to embed found.")
            return

        self.embeddings = self.embedder.generate_embeddings(
            self.df[col_exists].fillna('').agg(' '.join, axis=1).tolist()
        )
        self.logger.info(f"Embeddings generated. Shape: {self.embeddings.shape}")

    def process_pipeline(self, time_horizons):
        """
        Executes the full data processing pipeline:
        1) Clean price data + perform market analysis
        2) Preprocess news
        3) Asof-merge price + news
        4) Drop incomplete news rows
        5) Sentiment analysis
        6) Embedding generation
        :param time_horizons: list of time horizon dicts 
                              (we only use them to find max_gather_minutes).
        """
        self.logger.info("Starting data processing pipeline...")

        # 0) Validate time_horizons
        if not time_horizons:
            raise ValueError("time_horizons is empty. Ensure it is generated before processing.")
        max_gather_minutes = max(
            int(cfg['time_horizon'].total_seconds() // 60) 
            for cfg in time_horizons
        )

        # 1) Clean price + analyze
        self.clean_price_data()
        self.perform_market_analysis(max_gather_minutes, step=5)

        # 2) Preprocess news
        self.preprocess_news()

        # 3) Merge with asof logic (e.g. 5-minute tolerance)
        self.merge_data_asof(tolerance="5min", direction="backward")

        # 4) Drop incomplete news rows
        self.drop_incomplete_news()

        # 5) Sentiment
        self.process_sentiment()

        # 6) Embedding
        self.generate_embeddings(columns_to_embed=['title','summary'])

        if self.df is None:
            self.logger.warning("Final DataFrame is None.")
            return pd.DataFrame()

        self.logger.info(f"Final processed DataFrame shape: {self.df.shape}")
        return self.df
