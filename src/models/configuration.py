"""
Configuration Module

Defines the training configuration for model experiments using dataclasses.
This module provides a structure to define different experiment setups.
Additional fields can be added here, and defaults can be overridden via the centralized config.py.
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainingConfig:
    """
    Configuration for a single training experiment.
    """
    # Basic pipeline settings
    max_combos: int = 100
    save_best_only: bool = True

    # Sentiment filtering settings
    filter_sentiment: bool = False
    sentiment_threshold: float = 0.2
    sentiment_cols: List[str] = field(default_factory=lambda: ["title_positive", "summary_negative"])
    sentiment_mode: str = "any"  # "any" or "all"

    # Fluctuation filtering settings
    filter_fluctuation: bool = False
    fluct_threshold: float = 1.0  # For example, remove rows with |target| < 1.0

    # Model training hyperparameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])

def get_experiment_configurations() -> List[TrainingConfig]:
    """
    Returns a list of TrainingConfig objects for different experiments.
    These configurations are optional and can be used to run multiple training experiments.
    """
    return [
        TrainingConfig(
            max_combos=50,
            filter_sentiment=True,
            sentiment_threshold=0.3,
            sentiment_cols=["title_positive", "summary_negative", "expected_sentiment"],
            sentiment_mode="any",
            filter_fluctuation=False,
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            hidden_layers=[256, 128, 64]
        ),
        TrainingConfig(
            max_combos=50,
            filter_sentiment=True,
            sentiment_threshold=0.4,
            sentiment_cols=["summary_positive", "summary_negative"],
            sentiment_mode="all",
            filter_fluctuation=True,
            fluct_threshold=2.0,
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            hidden_layers=[256, 128, 64]
        ),
        # Additional configurations can be added as needed.
    ]
