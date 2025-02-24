# src/trading/trade_simulator.py

import os
import datetime
import pandas as pd
from src.utils.logger import get_logger
from src.trading.trading_config import trading_settings

class TradeSimulator:
    """
    Simulates trades based on trading decisions and logs the outcomes.
    """
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.trade_log = []
        self.initial_capital = trading_settings.initial_capital
        self.current_capital = self.initial_capital

    def execute_trade(self, decision: str, prediction: float, rationale: str):
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        trade_record = {
            "timestamp": timestamp,
            "decision": decision,
            "prediction": prediction,
            "rationale": rationale,
            "capital_before": self.current_capital
        }
        if decision == "BUY":
            profit = self.current_capital * (prediction / 100)
            self.current_capital += profit
            trade_record["profit"] = profit
        elif decision == "SELL":
            loss = self.current_capital * (abs(prediction) / 100)
            self.current_capital -= loss
            trade_record["loss"] = loss
        else:
            trade_record["profit"] = 0.0
        trade_record["capital_after"] = self.current_capital
        self.trade_log.append(trade_record)
        self.logger.info(f"Executed trade: {trade_record}")
        return trade_record

    def save_trade_log(self, filename: str = "trade_log.csv"):
        if not self.trade_log:
            self.logger.info("No trades to save.")
            return
        df = pd.DataFrame(self.trade_log)
        output_dir = trading_settings.permanent_storage_path
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        df.to_csv(file_path, index=False)
        self.logger.info(f"Trade log saved to {file_path}")

if __name__ == "__main__":
    simulator = TradeSimulator()
    test_trades = [("BUY", 1.5, "Test BUY trade"),
                   ("HOLD", 0.5, "Test HOLD trade"),
                   ("SELL", -2.0, "Test SELL trade")]
    for decision, prediction, rationale in test_trades:
        simulator.execute_trade(decision, prediction, rationale)
    simulator.save_trade_log()
