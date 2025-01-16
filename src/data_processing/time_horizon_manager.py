from datetime import timedelta
import numpy as np
from utils.logger import get_logger


class TimeHorizonManager:
    """
    Generates time horizon pairs for model testing.
    """

    def __init__(self, max_gather_minutes=2880, max_predict_minutes=10080):
        """
        :param max_gather_minutes: Maximum gather time in minutes.
        :param max_predict_minutes: Maximum prediction time in minutes.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.max_gather_minutes = max_gather_minutes
        self.max_predict_minutes = max_predict_minutes

    def generate_horizon_combos(self, step=5):
        """
        Dynamically generates gather-predict horizon pairs for training.
        :param step: Step size for intervals in minutes (default is 5).
        :return: List of dictionaries containing gather and predict horizons.
        """
        combos = []

        for g in range(step, self.max_gather_minutes + 1, step):
            min_predict_time = g + step
            max_predict_time = min(g * 2, self.max_predict_minutes)

            for p in range(min_predict_time, max_predict_time + 1, step):
                combos.append({
                    "gather_name": f"{g}_minutes",
                    "gather_td": timedelta(minutes=g),
                    "predict_name": f"{p}_minutes",
                    "predict_td": timedelta(minutes=p),
                })

        self.logger.info(f"Generated {len(combos)} horizon combinations.")
        return combos
