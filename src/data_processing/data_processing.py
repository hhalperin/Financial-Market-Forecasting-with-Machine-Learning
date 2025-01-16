import pandas as pd
import numpy as np
import talib
from utils.logger import get_logger
from data_processing.sentiment_processor import SentimentProcessor
from data_processing.market_analyzer import MarketAnalyzer
from data_processing.data_embedder import DataEmbedder

pd.set_option('future.no_silent_downcasting', True)

class DataProcessor:
    """
    Handles the end-to-end processing pipeline for merging, analyzing, and preparing data for ML.
    """

    def __init__(self, price_df, news_df, sentiment_model="ProsusAI/finbert", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
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
        - Converts 'time_published' to datetime and rounds to the nearest minute.
        """
        self.news_df['time_published'] = pd.to_datetime(self.news_df['time_published'], errors='coerce').dt.round('min')
        self.news_df.dropna(subset=['time_published'], inplace=True)
        self.news_df.sort_values('time_published', inplace=True)
        self.logger.info(f"Preprocessed news_df. Shape: {self.news_df.shape}")

    def perform_market_analysis(self, time_horizons):
        """
        Adds technical indicators and price fluctuations to the price DataFrame.
        :param time_horizons: List of dictionaries specifying time horizons.
        """
        analyzer = MarketAnalyzer(self.price_df)
        self.price_df = analyzer.analyze_market(time_horizons)
        self.logger.info(f"Market analysis completed. Shape: {self.price_df.shape}")

    def merge_data(self):
        """
        Merges price_df and news_df by aligning timestamps and filters to keep rows with valid articles.
        """
        # Ensure both DataFrames have 'DateTime' as the key
        self.price_df['DateTime'] = pd.to_datetime(self.price_df['DateTime'], errors='coerce').dt.round('min')
        self.news_df['time_published'] = pd.to_datetime(self.news_df['time_published'], errors='coerce').dt.round('min')

        # Rename for consistency before merging
        self.news_df.rename(columns={'time_published': 'DateTime'}, inplace=True)

        # Merge data
        merged_df = pd.merge(
            self.price_df,
            self.news_df,
            on='DateTime',
            how='outer',
            suffixes=('', '_news')
        )
        # Retain rows where news data exists
        merged_df_with_news = merged_df[
            (merged_df['title'].notna() & (merged_df['title'] != '')) |
            (merged_df['summary'].notna() & (merged_df['summary'] != ''))
        ]

        # Log the difference in shape after filtering
        self.logger.info(f"Merged data shape before filtering: {merged_df.shape}")
        self.logger.info(f"Merged data shape after filtering for articles: {merged_df_with_news.shape}")

        # Sort the merged DataFrame by DateTime
        merged_df_with_news = merged_df_with_news.sort_values('DateTime').copy()


        self.df = merged_df_with_news


    def process_sentiment(self):
        """
        Performs sentiment analysis on titles and summaries in the DataFrame.
        """
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

    def calculate_dynamic_targets(self, column_name, target_configs):
        """
        Creates dynamic target columns based on prediction time horizons.
        :param column_name: Column to base the target (e.g., 'Close').
        :param target_configs: List of configurations with prediction horizons.
        """
        for config in target_configs:
            target_name = config['target_name']
            predict_td = config['time_horizon']
            minutes = int(predict_td.total_seconds() // 60)
            self.df[target_name] = self.df[column_name].shift(-minutes) - self.df[column_name]
        
        # Fill missing values and ensure proper dtype inference
        self.df.fillna(0, inplace=True)
        self.logger.info("Dynamic targets calculated.")

    def generate_embeddings(self, columns_to_embed=None):
        """
        Generates embeddings for specified columns in the DataFrame.
        :param columns_to_embed: List of column names to embed.
        """
        if self.df is None or self.df.empty:
            self.logger.warning("No DataFrame available to generate embeddings.")
            return

        self.embeddings = self.embedder.generate_embeddings(
            self.df[columns_to_embed].fillna('').agg(' '.join, axis=1).tolist()
        )
        self.logger.info(f"Embeddings generated. Shape: {self.embeddings.shape}")

    def process_pipeline(self, time_horizons, target_configs):
        """
        Executes the full data processing pipeline.
        - Cleans price data.
        - Performs market analysis.
        - Preprocesses news data.
        - Merges data.
        - Performs sentiment analysis.
        - Generates embeddings.
        - Calculates dynamic targets.
        :param time_horizons: List of time horizon configurations.
        :param target_configs: List of target configurations.
        :return: Processed DataFrame.
        """
        self.logger.info("Starting data processing pipeline...")

        # Step 1: Clean and analyze price data
        self.clean_price_data()
        self.perform_market_analysis(time_horizons)

        # Step 2: Preprocess and merge news data
        self.preprocess_news()
        self.merge_data()

        # Step 3: Perform sentiment analysis
        self.process_sentiment()

        # Step 4: Generate embeddings
        self.generate_embeddings(columns_to_embed=['title', 'summary'])

        # Step 5: Calculate dynamic targets
        self.calculate_dynamic_targets('Close', target_configs)

        self.logger.info(f"Final processed DataFrame shape: {self.df.shape}")
        return self.df