"""
Sentiment Processor Module

Performs sentiment analysis on news texts and computes a rolling expected sentiment based on past sentiments
and reactions. Optionally applies recency weighting using a decay factor.
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from typing import Tuple, List, Any
from src.utils.logger import get_logger
import math

class SentimentProcessor:
    def __init__(self, model_name: str = "ProsusAI/finbert", 
                 use_recency_weighting: bool = True,
                 recency_decay: float = 0.01) -> None:
        """
        Initializes the SentimentProcessor.

        :param model_name: Hugging Face model for sentiment analysis.
        :param use_recency_weighting: Whether to weight past sentiments by recency.
        :param recency_decay: Decay factor (per minute) for recency weighting.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.model_name = model_name
        self.use_recency_weighting = use_recency_weighting
        self.recency_decay = recency_decay
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"SentimentProcessor initialized with model: {model_name}")

    def analyze_sentiment(self, texts: List[str], batch_size: int = 16) -> Tuple[List[float], List[float], List[float], List[str]]:
        """
        Analyzes sentiment for a list of texts in batches.

        :param texts: List of texts to analyze.
        :param batch_size: Batch size for processing.
        :return: Tuple of (positive_probs, negative_probs, neutral_probs, predicted_labels).
        """
        positive_probs, negative_probs, neutral_probs, predicted_labels = [], [], [], []
        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Batches"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            for prob in probs:
                positive_probs.append(prob[0])
                negative_probs.append(prob[1])
                neutral_probs.append(prob[2])
                predicted_labels.append(self.model.config.id2label[prob.argmax()])
        return positive_probs, negative_probs, neutral_probs, predicted_labels

    def compute_expected_sentiment(self, df: pd.DataFrame, sentiment_col: str = "summary_sentiment", reaction_col: str = "minutes_change") -> pd.DataFrame:
        """
        Computes an expected sentiment for each row using a rolling window of past sentiments and reactions.
        Applies recency weighting if enabled.

        :param df: DataFrame containing 'DateTime', sentiment_col, and reaction_col.
        :param sentiment_col: Column name for sentiment scores.
        :param reaction_col: Column name for reaction magnitudes.
        :return: DataFrame with a new 'expected_sentiment' column.
        """
        if sentiment_col not in df.columns:
            self.logger.warning(f"Column '{sentiment_col}' not found; skipping expected sentiment calculation.")
            df["expected_sentiment"] = 0
            return df

        if "DateTime" not in df.columns:
            self.logger.error("Column 'DateTime' not found. Cannot compute expected sentiment.")
            return df

        df.sort_values("DateTime", inplace=True)
        past_sentiments, past_reactions, past_datetimes, expected_sentiments = [], [], [], []
        window_size = 5  # Use the last 5 records.
        for idx, row in df.iterrows():
            current_time = row["DateTime"]
            cur_sentiment = self._safe_float(row.get(sentiment_col, 0))
            cur_reaction = abs(self._safe_float(row.get(reaction_col, 1)))
            current_datetime = pd.to_datetime(current_time)
            past_sentiments.append(cur_sentiment)
            past_reactions.append(cur_reaction)
            past_datetimes.append(current_datetime)
            if len(past_sentiments) > window_size:
                past_sentiments.pop(0)
                past_reactions.pop(0)
                past_datetimes.pop(0)
            if self.use_recency_weighting and past_datetimes:
                weights = [reaction * math.exp(-self.recency_decay * ((current_datetime - past_time).total_seconds() / 60.0))
                           for past_time, reaction in zip(past_datetimes, past_reactions)]
                total_weight = sum(weights)
                expected = sum(s * w for s, w in zip(past_sentiments, weights)) / total_weight if total_weight > 0 else 0
            else:
                total_weight = sum(past_reactions)
                expected = sum(s * w for s, w in zip(past_sentiments, past_reactions)) / total_weight if total_weight > 0 else 0
            expected_sentiments.append(expected)
        df["expected_sentiment"] = expected_sentiments
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """
        Safely converts a value to float.

        :param value: Value to convert.
        :param default: Default value if conversion fails.
        :return: Float value.
        """
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
