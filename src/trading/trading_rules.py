# src/trading/trading_rules.py

from src.utils.logger import get_logger

class TradingRules:
    """
    Contains trading rules logic to decide whether to BUY, SELL, or HOLD based on model predictions.
    """
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.buy_threshold = 1.0    # percent threshold to trigger BUY
        self.sell_threshold = -1.0  # percent threshold to trigger SELL

    def evaluate_trade(self, predicted_fluctuation: float):
        if predicted_fluctuation >= self.buy_threshold:
            decision = "BUY"
            rationale = f"Predicted increase of {predicted_fluctuation:.2f}% exceeds buy threshold of {self.buy_threshold}%."
        elif predicted_fluctuation <= self.sell_threshold:
            decision = "SELL"
            rationale = f"Predicted decrease of {predicted_fluctuation:.2f}% exceeds sell threshold of {self.sell_threshold}%."
        else:
            decision = "HOLD"
            rationale = f"Predicted fluctuation of {predicted_fluctuation:.2f}% does not trigger any trade action."
        self.logger.info(f"Evaluated trade decision: {decision} | {rationale}")
        return decision, rationale
