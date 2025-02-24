# src/trading/trading_engine.py

import os
import json
import torch
import numpy as np
import pandas as pd
from src.utils.logger import get_logger
from src.trading.trading_config import trading_settings
from src.trading.trading_rules import TradingRules
from src.trading.trade_simulator import TradeSimulator
from src.models.stock_predictor import StockPredictor

def prepare_features(numeric_data):
    """
    Converts numeric_data (loaded from .npy) into a 2D float32 array.
    It creates a DataFrame, selects only numeric columns, and returns a proper float32 NumPy array.
    """
    df = pd.DataFrame(numeric_data)
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.empty:
        get_logger("prepare_features").error("No numeric columns found in processed data.")
        return np.empty((df.shape[0], 0))
    return df_numeric.to_numpy(dtype=np.float32)

class TradingEngine:
    def __init__(self, ticker: str, local_mode: bool = True):
        self.ticker = ticker
        self.local_mode = local_mode
        self.logger = get_logger(self.__class__.__name__)
        self.model = None
        self.trading_rules = TradingRules()
        self.trade_simulator = TradeSimulator()
        self.load_model()
    
    def load_model(self):
        """
        Loads the pre-trained model based on goated model info.
        Fixes the checkpoint path if the goated info already includes the prefix.
        """
        try:
            goated_info_path = os.path.join(
                trading_settings.goated_models_dir,
                "goated_directional_accuracy",
                "goated_directional_accuracy_info.json"
            )
            if os.path.exists(goated_info_path):
                self.logger.info(f"Loading goated model info from {goated_info_path}")
                with open(goated_info_path, 'r') as f:
                    goated_info = json.load(f)
                architecture = goated_info.get("architecture", [128, 64, 32])
                model_name_from_json = goated_info.get("model_name", "")
                dest_stage = goated_info.get("dest_stage", "")
                self.logger.info(f"Goated info: model_name={model_name_from_json}, dest_stage={dest_stage}, architecture={architecture}")
                # Remove redundant prefix if present:
                prefix = f"models{os.sep}best_models"
                if dest_stage.startswith(prefix):
                    relative_path = dest_stage[len(prefix):].lstrip(os.sep)
                else:
                    relative_path = dest_stage
                checkpoint_path = os.path.join(trading_settings.best_models_dir, relative_path, "best_model.pt")
                self.logger.info(f"Resolved checkpoint path from goated info: {checkpoint_path}")
            else:
                self.logger.info("No goated model info found; falling back to default best model.")
                architecture = [256, 128, 64]
                checkpoint_path = os.path.join(trading_settings.best_models_dir, f"{self.ticker}_best_model.pt")
                self.logger.info(f"Using fallback checkpoint path: {checkpoint_path}")
            
            # For prediction, input_size should match your training configuration.
            # Here we assume the composite embedding plus any additional numeric features gives input_size=34.
            input_size = 34
            self.model = StockPredictor(input_size, architecture).to(torch.device("cpu"))
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.logger.info(f"Successfully loaded model from {checkpoint_path} with architecture {architecture}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.model = None

    def process_and_predict(self, price_df: pd.DataFrame, news_df: pd.DataFrame, processed_array=None) -> float:
        """
        Uses the processed numeric data to produce a prediction.
        If processed_array is provided, it is used directly.
        """
        self.logger.info(f"Received price_df with shape {price_df.shape}")
        self.logger.info(f"Received news_df with shape {news_df.shape}")
        
        if processed_array is not None and processed_array.size > 0:
            self.logger.info("Using cached numeric data for prediction.")
            features = prepare_features(processed_array)
            if features is None or features.size == 0:
                self.logger.warning("Prepared features are empty.")
                return 0.0
            # Use the last row for prediction
            features_tensor = torch.tensor(features[-1:], dtype=torch.float32)
            with torch.no_grad():
                prediction = self.model(features_tensor).item()
            self.logger.info(f"Predicted expected price fluctuation: {prediction:.4f}%")
            return prediction
        
        self.logger.error("No processed numeric data provided; cannot predict.")
        return 0.0

    def execute_trading_cycle(self, price_df: pd.DataFrame, news_df: pd.DataFrame, processed_array=None):
        prediction = self.process_and_predict(price_df, news_df, processed_array)
        decision, rationale = self.trading_rules.evaluate_trade(prediction)
        self.logger.info(f"Trading Decision: {decision} | Rationale: {rationale}")
        self.trade_simulator.execute_trade(decision, prediction, rationale)
        return decision, prediction, rationale

if __name__ == "__main__":
    import pandas as pd
    engine = TradingEngine(ticker="NVDA", local_mode=True)
    sample_price_path = os.path.join(trading_settings.data_storage_path, "raw", "NVDA_raw_price_2025-02-18.csv")
    sample_news_path = os.path.join(trading_settings.data_storage_path, "raw", "NVDA_raw_news_2025-02-18.csv")
    if os.path.exists(sample_price_path) and os.path.exists(sample_news_path):
        price_df = pd.read_csv(sample_price_path)
        news_df = pd.read_csv(sample_news_path)
        engine.execute_trading_cycle(price_df, news_df)
    else:
        engine.logger.error("Sample raw data files not found for testing.")
