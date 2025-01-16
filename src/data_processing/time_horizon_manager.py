from datetime import timedelta
import numpy as np
from utils.logger import get_logger


class TimeHorizonManager:
    """
    Generates time horizon pairs for model testing.
    - Gather time is always less than the prediction time.
    - Both gather and prediction times are divisible by a given step.
    """

    def __init__(self, max_gather_minutes=2880, max_predict_minutes=10080):
        """
        :param max_gather_minutes: Maximum gather time in minutes.
        :param max_predict_minutes: Maximum prediction time in minutes.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.max_gather_minutes = max_gather_minutes
        self.max_predict_minutes = max_predict_minutes

    def testing_horizon_combos(self, step=5, max_combos=10):
        """
        Generates spaced-out time horizon combos for testing. Ensures divisibility by `step` and uses Python integers.
        """
        gather_values = np.linspace(step, self.max_gather_minutes, num=max_combos, endpoint=True, dtype=int)
        combos = []

        for g in gather_values:
            g = int(g - (g % step))  # Ensure divisibility by `step`
            gather_td = timedelta(minutes=g)
            predict_td = timedelta(minutes=int(min(g * 2, self.max_predict_minutes) - min(g * 2, self.max_predict_minutes) % step))
            combos.append({
                "gather_name": f"{g}_minutes",
                "gather_td": gather_td,
                "predict_name": f"{int(predict_td.total_seconds() // 60)}_minutes",
                "predict_td": predict_td
            })
        return combos

    def generate_horizon_combos(self, step=5, max_combos=None):
        """
        Dynamically generates key-value time horizon pairs with scaling gather and prediction times.
        """
        gather_values = np.arange(step, self.max_gather_minutes + step, step)
        combos = []

        for g in gather_values:
            g = g - (g % step)  # Ensure divisibility by `step`
            min_predict_time = g + step
            max_predict_time = min(g * 2, self.max_predict_minutes)
            predict_values = np.arange(min_predict_time, max_predict_time + step, step)

            for p in predict_values:
                p = p - (p % step)  # Ensure divisibility by `step`
                combos.append({
                    "gather_name": f"{g}_minutes",
                    "gather_td": timedelta(minutes=g),
                    "predict_name": f"{p}_minutes",
                    "predict_td": timedelta(minutes=p)
                })
                if max_combos and len(combos) >= max_combos:
                    return combos

        return combos
