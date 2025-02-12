"""
Data Processing Module

This module provides classes for processing aggregated data:
- DataProcessor: End-to-end pipeline for cleaning, merging, analyzing, and embedding data.
- DataEmbedder: Generates embeddings from text data using Hugging Face models.
- SentimentProcessor: Performs sentiment analysis and computes expected sentiment.
- MarketAnalyzer: Computes technical indicators and price fluctuations.
- TimeHorizonManager: Generates time horizon combinations for model training.
"""

from .data_processing import DataProcessor
from .data_embedder import DataEmbedder
from .sentiment_processor import SentimentProcessor
from .market_analyzer import MarketAnalyzer
from .time_horizon_manager import TimeHorizonManager

__all__ = [
    "DataProcessor",
    "DataEmbedder",
    "SentimentProcessor",
    "MarketAnalyzer",
    "TimeHorizonManager"
]
