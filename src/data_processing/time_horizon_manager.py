"""
Time Horizon Manager Module

Generates combinations of time horizons for model training and testing.
This updated version uses a multiplier factor to determine prediction times relative to gather times.
For each gather time 'g', the prediction time 'p' is computed as:
    p = int(round(g * multiplier))
where multiplier varies from multiplier_min to multiplier_max in steps of multiplier_step.
For gather times within the first day (<= 1440 minutes), the multiplier range is restricted
to yield more consistent prediction horizons.
Duplicate combinations are removed and the resulting combos are sorted.
"""

from datetime import timedelta
from typing import List, Dict, Any
from src.utils.logger import get_logger

class TimeHorizonManager:
    def __init__(
        self,
        max_gather_minutes: int = 10080,
        max_predict_minutes: int = 40320,
        step: int = 1,
        multiplier_min: float = 1.1,
        multiplier_max: float = 8.0,
        multiplier_step: float = 0.1
    ) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.max_gather_minutes = max_gather_minutes
        self.max_predict_minutes = max_predict_minutes
        self.step = step
        self.multiplier_min = multiplier_min
        self.multiplier_max = multiplier_max
        self.multiplier_step = multiplier_step

    def generate_horizon_combos(self) -> List[Dict[str, Any]]:
        """
        Generates unique horizon combinations based on gather and prediction times.
        
        :return: Sorted list of horizon combination dictionaries.
        """
        combos_dict = {}
        for g in range(self.step, self.max_gather_minutes + 1, self.step):
            # For gather times within the first day (1440 minutes), restrict the multiplier range
            if g <= 1440:
                local_multiplier_max = min(self.multiplier_max, self.multiplier_min + 1.0)
            else:
                local_multiplier_max = self.multiplier_max

            multiplier = self.multiplier_min
            while multiplier <= local_multiplier_max:
                p = int(round(g * multiplier))
                if p > g and p <= self.max_predict_minutes:
                    key = (g, p)
                    if key not in combos_dict:
                        combos_dict[key] = {
                            "gather_name": f"{g}_minutes",
                            "gather_td": timedelta(minutes=g),
                            "predict_name": f"{p}_minutes",
                            "predict_td": timedelta(minutes=p),
                            "multiplier": multiplier
                        }
                multiplier += self.multiplier_step
        combos = list(combos_dict.values())
        combos.sort(key=lambda combo: (combo["gather_td"].total_seconds(), combo["predict_td"].total_seconds()))
        self.logger.info(f"Generated {len(combos)} unique horizon combinations.")
        return combos

    def filter_combos(self, combos: List[Dict[str, Any]], max_combos: int = 20000) -> List[Dict[str, Any]]:
        """
        Uniformly samples from the list of horizon combos if the total exceeds max_combos.

        :param combos: List of horizon combination dictionaries.
        :param max_combos: Maximum number of combos to keep.
        :return: Filtered list of combos.
        """
        total = len(combos)
        if total <= max_combos:
            self.logger.info(f"Using all {total} horizon combinations.")
            return combos
        step_size = total / max_combos
        filtered_combos = []
        for i in range(max_combos):
            index = int(i * step_size)
            filtered_combos.append(combos[index])
        self.logger.info(f"Filtered to {len(filtered_combos)} horizon combinations (uniformly sampled).")
        return filtered_combos
