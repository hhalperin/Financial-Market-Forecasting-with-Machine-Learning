from datetime import timedelta
import numpy as np
from src.utils.logger import get_logger


class TimeHorizonManager:
    """
    Generates time horizon pairs for model training and testing.
    """

    def __init__(self, max_gather_minutes=2880, max_predict_minutes=10080, step=5):
        """
        :param max_gather_minutes: Maximum gather time in minutes.
        :param max_predict_minutes: Maximum prediction time in minutes.
        :param step: Step size for intervals in minutes.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.max_gather_minutes = max_gather_minutes
        self.max_predict_minutes = max_predict_minutes
        self.step = step

    def generate_horizon_combos(self):
        """
        Generate gather-predict horizon combinations.
        :return: List of dictionaries with gather and predict horizons.
        """
        combos = []

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

    def filter_combos(self, combos, max_combos=2000):
        """0
        Optionally limit the number of horizon combos.
        :param combos: List of generated horizon combinations.
        :param max_combos: Maximum number of combinations to retain.
        :return: Filtered list of combos.
        """
        filtered_combos = combos[:max_combos]
        self.logger.info(f"Filtered to {len(filtered_combos)} horizon combinations.")
        return filtered_combos
