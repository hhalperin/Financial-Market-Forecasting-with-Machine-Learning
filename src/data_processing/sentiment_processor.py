"""
Sentiment Processor Module

Performs sentiment analysis on news texts and computes a rolling expected sentiment based on past reactions.
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from typing import Tuple, List, Any
from src.utils.logger import get_logger

class SentimentProcessor:
    """
    Analyzes sentiment of news articles using a Hugging Face model and computes a rolling expected sentiment.
    """
    def __init__(self, model_name: str = 'ProsusAI/finbert') -> None:
        """
        Initializes the SentimentProcessor.

        :param model_name: Hugging Face model for sequence classification.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"SentimentProcessor initialized with model: {model_name}")

    def analyze_sentiment(self, texts: List[str], batch_size: int = 16) -> Tuple[List[float], List[float], List[float], List[str]]:
        """
        Performs sentiment analysis on a list of texts.

        :param texts: List of news article texts.
        :param batch_size: Number of texts per batch.
        :return: Tuple containing lists of positive, negative, neutral probabilities and predicted labels.
        """
        positive_probs, negative_probs, neutral_probs, predicted_labels = [], [], [], []
        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Batches"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
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

    def compute_expected_sentiment(self, df: pd.DataFrame, sentiment_col: str = 'summary_sentiment', reaction_col: str = 'minutes_change') -> pd.DataFrame:
        """
        Computes a rolling expected sentiment based on past sentiments and market reactions.

        :param df: DataFrame containing sentiment and market reaction columns.
        :param sentiment_col: Column name containing sentiment scores.
        :param reaction_col: Column name containing market reaction values.
        :return: DataFrame with an added 'expected_sentiment' column.
        """
        if sentiment_col not in df.columns:
            self.logger.warning(f"Column '{sentiment_col}' not found. Skipping expected sentiment calculation.")
            df['expected_sentiment'] = 0
            return df

        if 'DateTime' not in df.columns:
            self.logger.error("Column 'DateTime' not found. Cannot compute expected sentiment.")
            return df

        df.sort_values(by='DateTime', inplace=True)
        past_sentiments, past_reactions, sentiments = [], [], []
        for idx, row in df.iterrows():
            cur_sentiment = self._safe_float(row.get(sentiment_col, 0))
            cur_reaction = abs(self._safe_float(row.get(reaction_col, 1)))
            past_sentiments.append(cur_sentiment)
            past_reactions.append(cur_reaction)
            if len(past_sentiments) > 5:
                past_sentiments.pop(0)
                past_reactions.pop(0)
            total_weight = sum(past_reactions)
            expected_sentiment = (sum(s * w for s, w in zip(past_sentiments, past_reactions)) / total_weight) if total_weight > 0 else 0
            sentiments.append(expected_sentiment)
        df['expected_sentiment'] = sentiments
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """
        Safely converts a value to float, returning a default on error.
        """
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
