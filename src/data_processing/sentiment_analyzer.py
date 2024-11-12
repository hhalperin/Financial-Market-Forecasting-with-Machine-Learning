import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import logging
from utils.logger import get_logger

class SentimentAnalyzer:
    """
    Performs sentiment analysis using FinBERT.
    """

    def __init__(self, model_name='ProsusAI/finbert'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.logger = get_logger(self.__class__.__name__)

    def analyze_sentiment(self, texts, batch_size=16):
        """
        Analyzes sentiment for a list of texts.
        """
        positive_probs, negative_probs, neutral_probs, predicted_labels = [], [], [], []

        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analysis"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()

            for prob in probabilities:
                positive_probs.append(prob[0])
                negative_probs.append(prob[1])
                neutral_probs.append(prob[2])
                predicted_label_id = prob.argmax()
                predicted_label = self.model.config.id2label[predicted_label_id]
                predicted_labels.append(predicted_label)

        self.logger.info("Completed sentiment analysis.")
        return positive_probs, negative_probs, neutral_probs, predicted_labels
