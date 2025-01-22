# src/models/configuration.py

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TrainingConfig:
    """
    Defines the configuration for a single run of the model pipeline.
    """
    # Basic pipeline settings
    max_combos: int = 100
    save_best_only: bool = True

    # Sentiment filtering
    filter_sentiment: bool = False
    sentiment_threshold: float = 0.2
    sentiment_cols: List[str] = field(default_factory=lambda: ["title_positive", "summary_negative"])
    sentiment_mode: str = "any"  # or "all"

    # Fluctuation filtering
    filter_fluctuation: bool = False
    fluct_threshold: float = 1.0  # e.g., remove rows w/ |target_col| < 1.0

    # Possibly other settings
    # e.g., how many epochs or batch size if you want them here
    # we can keep it minimal since we already set some in ModelManager if we want

def get_experiment_configurations():
    """
    Returns a list of TrainingConfig objects, each describing a different experiment.
    """
    return [
        TrainingConfig(
            max_combos=50,
            filter_sentiment=True,
            sentiment_threshold=0.3,
            sentiment_cols=["title_positive", "summary_negative", "expected_sentiment"],
            sentiment_mode="any",
            filter_fluctuation=False
        ),
        TrainingConfig(
            max_combos=50,
            filter_sentiment=True,
            sentiment_threshold=0.4,
            sentiment_cols=["summary_positive", "summary_negative"],
            sentiment_mode="all",
            filter_fluctuation=True,
            fluct_threshold=2.0
        ),
        # Add as many different combos as you want...
    ]
