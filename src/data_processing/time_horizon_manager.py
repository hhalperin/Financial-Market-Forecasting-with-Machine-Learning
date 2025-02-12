"""
Time Horizon Manager Module

Generates combinations of time horizons for model training and testing.
"""

from datetime import timedelta
from typing import List, Dict, Any
from src.utils.logger import get_logger

class TimeHorizonManager:
    """
    Generates gather-predict horizon combinations.
    """
    def __init__(self, max_gather_minutes: int = 2880, max_predict_minutes: int = 10080, step: int = 5) -> None:
        """
        Initializes the TimeHorizonManager.

        :param max_gather_minutes: Maximum gather time in minutes.
        :param max_predict_minutes: Maximum prediction time in minutes.
        :param step: Step size (in minutes) for intervals.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.max_gather_minutes = max_gather_minutes
        self.max_predict_minutes = max_predict_minutes
        self.step = step

    def generate_horizon_combos(self) -> List[Dict[str, Any]]:
        """
        Generates combinations of gathering and prediction time horizons.

        :return: List of dictionaries with keys 'gather_name', 'gather_td', 'predict_name', and 'predict_td'.
        """
        combos: List[Dict[str, Any]] = []
        for g in range(self.step, self.max_gather_minutes + 1, self.step):
            min_predict_time = g + self.step
            max_predict_time = min(g * 2, self.max_predict_minutes)
            for p in range(min_predict_time, max_predict_time + 1, self.step):
                combos.append({
                    "gather_name": f"{g}_minutes",
                    "gather_td": timedelta(minutes=g),
                    "predict_name": f"{p}_minutes",
                    "predict_td": timedelta(minutes=p),
                })
        self.logger.info(f"Generated {len(combos)} horizon combinations.")
        return combos

    def filter_combos(self, combos: List[Dict[str, Any]], max_combos: int = 2000) -> List[Dict[str, Any]]:
        """
        Optionally limits the number of horizon combinations.

        :param combos: List of generated horizon combinations.
        :param max_combos: Maximum number of combinations to retain.
        :return: Filtered list of horizon combinations.
        """
        filtered_combos = combos[:max_combos]
        self.logger.info(f"Filtered to {len(filtered_combos)} horizon combinations.")
        return filtered_combos
