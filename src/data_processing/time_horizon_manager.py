"""
Time Horizon Manager Module

Generates combinations of time horizons for model training and testing.
This updated version uses a multiplier factor to determine prediction times relative to gather times.
For each gather time 'g', the prediction time 'p' is computed as p = int(round(g * multiplier)),
where multiplier varies from multiplier_min to multiplier_max in steps of multiplier_step.
Duplicate combinations (based on gather and prediction times) are removed.
The resulting combos are then sorted by gather time and prediction time,
ensuring that later uniform sampling produces a balanced subset.
"""

from datetime import timedelta
from typing import List, Dict, Any
from src.utils.logger import get_logger
import random

class TimeHorizonManager:
    """
    Generates gather-predict horizon combinations.
    """
    def __init__(
        self,
        max_gather_minutes: int = 10080,
        max_predict_minutes: int = 40320,
        step: int = 1,
        multiplier_min: float = 1.1,
        multiplier_max: float = 8.0,
        multiplier_step: float = 0.1
    ) -> None:
        """
        Initializes the TimeHorizonManager.
        
        :param max_gather_minutes: Maximum gather time in minutes.
        :param max_predict_minutes: Maximum prediction time in minutes.
        :param step: Step size (in minutes) for the gather times.
        :param multiplier_min: Minimum multiplier for prediction time relative to gather time.
        :param multiplier_max: Maximum multiplier for prediction time relative to gather time.
        :param multiplier_step: Step size for multiplier increments.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.max_gather_minutes = max_gather_minutes
        self.max_predict_minutes = max_predict_minutes
        self.step = step
        self.multiplier_min = multiplier_min
        self.multiplier_max = multiplier_max
        self.multiplier_step = multiplier_step

    def generate_horizon_combos(self) -> List[Dict[str, Any]]:
        """
        Generates unique combinations of gathering and prediction time horizons using a multiplier approach.
        
        For each gather time 'g' (in minutes) from 'step' up to max_gather_minutes, iterate over multipliers 
        from multiplier_min to multiplier_max (inclusive) in steps of multiplier_step. The prediction time 'p' is:
            p = int(round(g * multiplier))
        Only include combos where p > g and p <= max_predict_minutes.
        Duplicate (g, p) pairs are removed.
        
        :return: Sorted list of dictionaries with keys:
                 'gather_name' (e.g., "30_minutes"),
                 'gather_td' (timedelta for g),
                 'predict_name' (e.g., "150_minutes"),
                 'predict_td' (timedelta for p),
                 and 'multiplier' (the factor used).
        """
        combos_dict = {}
        for g in range(self.step, self.max_gather_minutes + 1, self.step):
            multiplier = self.multiplier_min
            while multiplier <= self.multiplier_max:
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
        # Sort combos first by gather time then by prediction time (both in seconds)
        combos.sort(key=lambda combo: (combo["gather_td"].total_seconds(), combo["predict_td"].total_seconds()))
        self.logger.info(f"Generated {len(combos)} unique horizon combinations.")
        return combos

    def filter_combos(self, combos: List[Dict[str, Any]], max_combos: int = 20000) -> List[Dict[str, Any]]:
        """
        Limits the number of horizon combinations by uniformly sampling from the sorted list.
        
        Because the combos are now sorted, uniformly sampling by index will yield an even distribution
        across the entire range.
        
        Alternatively, you could shuffle the list:
            random.shuffle(combos)
        and then take the first max_combos.
        
        :param combos: Sorted list of unique horizon combinations.
        :param max_combos: Maximum number of combinations to retain.
        :return: Filtered list of horizon combinations, uniformly sampled.
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
