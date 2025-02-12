"""
Model Pipeline Module

Coordinates data preparation and training across multiple time horizon combinations.
Includes sentiment filtering, dropping of future horizon columns, and saving of best models.
"""

import os
import shutil
from tqdm import tqdm
from src.utils.logger import get_logger
import pandas as pd
from typing import Any, Optional

class ModelPipeline:
    """
    Coordinates the model training process over different horizon combinations.
    """
    def __init__(self, model_manager: Any, data_handler: Any, horizon_manager: Any) -> None:
        """
        :param model_manager: Instance of ModelManager for training/evaluation.
        :param data_handler: DataHandler instance for saving artifacts.
        :param horizon_manager: Time horizon manager.
        """
        self.model_manager = model_manager
        self.data_handler = data_handler
        self.horizon_manager = horizon_manager
        self.logger = get_logger(self.__class__.__name__)

    def advanced_filter_sentiment(self, df: pd.DataFrame, sentiment_columns: list = None,
                                  threshold: float = 0.2, mode: str = "any") -> pd.DataFrame:
        """
        Filters the DataFrame based on sentiment columns.
        """
        if not sentiment_columns:
            self.logger.warning("No sentiment columns specified for filtering. Returning df unchanged.")
            return df
        missing_cols = [col for col in sentiment_columns if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing sentiment columns {missing_cols}.")
        valid_cols = [col for col in sentiment_columns if col in df.columns]
        if not valid_cols:
            self.logger.warning("No valid sentiment columns found. Returning df unchanged.")
            return df
        orig_len = len(df)
        bool_masks = [(df[col] >= threshold) for col in valid_cols]
        if mode == "any":
            combined_mask = bool_masks[0]
            for m in bool_masks[1:]:
                combined_mask = combined_mask | m
        else:
            combined_mask = bool_masks[0]
            for m in bool_masks[1:]:
                combined_mask = combined_mask & m
        filtered_df = df[combined_mask].copy()
        self.logger.info(f"Advanced sentiment filter: reduced rows from {orig_len} to {len(filtered_df)} "
                         f"(mode='{mode}', threshold={threshold}, columns={valid_cols}).")
        return filtered_df

    def _drop_future_horizon_columns(self, X: pd.DataFrame, gather_horizon_str: str) -> pd.DataFrame:
        """
        Drops columns corresponding to horizons that exceed the current gathering horizon.
        """
        gather_mins = int(gather_horizon_str.split("_minutes")[0])
        
        if not hasattr(X, "columns"):
            return X
        keep_cols = []
        for col in X.columns:
            if "_minutes" in col:
                try:
                    col_mins = int(col.split("_minutes")[0])
                    if col_mins <= gather_mins:
                        keep_cols.append(col)
                except ValueError:
                    keep_cols.append(col)
            else:
                keep_cols.append(col)
        X_dropped = X[keep_cols]
        dropped_count = X.shape[1] - X_dropped.shape[1]
        self.logger.info(f"Dropped {dropped_count} future-horizon columns for gather horizon {gather_horizon_str}. "
                         f"Remaining features: {X_dropped.shape[1]}.")
        return X_dropped

    def train_on_horizons(
        self,
        X: pd.DataFrame,
        df: pd.DataFrame,
        max_combos: int = 100,
        save_best_only: bool = True,
        filter_sentiment: bool = False,
        sentiment_threshold: float = 0.35,
        sentiment_cols: list = None,
        sentiment_mode: str = "any",
        combos: Optional[list] = None  # New optional parameter
    ) -> list:
        """
        Trains models over multiple time horizon combinations.
        Optionally applies sentiment filtering before training.
        """
        if filter_sentiment and sentiment_cols:
            df = self.advanced_filter_sentiment(df, sentiment_columns=sentiment_cols,
                                                threshold=sentiment_threshold, mode=sentiment_mode)
            X = X.loc[df.index]
        if combos is None:
            combos = self.horizon_manager.generate_horizon_combos()[:max_combos]
        results = []
        best_mse = float("inf")
        best_info = None
        best_training_data_path = None
        for combo in tqdm(combos, desc="Training across horizons", unit="horizon"):
            gather = combo["gather_name"]
            predict = combo["predict_name"]
            target_col = f"{predict}_percentage_change"
            if target_col not in df.columns:
                self.logger.warning(f"Missing target column '{target_col}' for horizon {predict}. Skipping.")
                continue
            df_f = df.dropna(subset=[target_col])
            if df_f.empty:
                continue
            X_f = X.loc[df_f.index]
            y = df_f[target_col].values
            if X_f.empty:
                continue
            X_f = self._drop_future_horizon_columns(X_f, gather)
            X_f_numeric = X_f.select_dtypes(include=["number"])
            if X_f_numeric.shape[1] == 0:
                self.logger.warning("No numeric features remain after dropping future horizon columns. Skipping this horizon.")
                continue
            self.logger.info(f"Training with {X_f_numeric.shape[1]} numeric features: {list(X_f_numeric.columns)}. Target Column: {target_col}")
            model_name = f"model_{gather}_to_{predict}"
            self.logger.info(f"Training model: {model_name}")
            old_stage = self.model_manager.model_stage
            horizon_stage = f"{old_stage}/best_models/{gather}_to_{predict}"
            self.model_manager.model_stage = horizon_stage
            old_input_size = self.model_manager.input_size
            self.model_manager.input_size = X_f_numeric.shape[1]
            model, metrics = self.model_manager.train_and_evaluate(X_f_numeric.values, y, model_name)
            self.model_manager.input_size = old_input_size
            self.model_manager.model_stage = old_stage
            if model is None:
                continue
            if save_best_only:
                best_dir = self._resolve_local_path(horizon_stage)
                os.makedirs(best_dir, exist_ok=True)
                model_path = os.path.join(best_dir, f"{model_name}.pt")
                self.model_manager.save_model(model, model_path)
                train_df = X_f_numeric.copy()
                train_df[target_col] = y
                training_data_filename = f"{model_name}_training_data.csv"
                self.data_handler.save_dataframe(train_df, training_data_filename, stage=horizon_stage)
                self.logger.info(f"Saved training data to {training_data_filename} in stage {horizon_stage}.")
                if metrics["mse"] < best_mse:
                    best_mse = metrics["mse"]
                    best_info = {"model_name": model_name, "stage": horizon_stage, "gather": gather, "predict": predict, "r2": metrics["r2"]}
                    best_training_data_path = os.path.join(best_dir, training_data_filename)
                    self.logger.info(f"Best model so far: {gather} to {predict} with R²: {metrics['r2']:.4f}")
            results.append({
                "gather_horizon": gather,
                "predict_horizon": predict,
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"]
            })
        if not results:
            self.logger.warning("No horizon combinations produced any results.")
            return results
        self._save_summary(results)
        if best_info:
            self._save_goated_model(best_info, best_training_data_path)
        return results

    def _resolve_local_path(self, stage_name: str) -> str:
        from src.utils.data_handler import DataHandler
        if self.data_handler.storage_mode == "s3":
            return os.path.join(self.data_handler.base_data_dir, stage_name)
        else:
            return os.path.join(self.data_handler.base_data_dir, stage_name)

    def _save_goated_model(self, best_info: dict, best_training_data_path: str) -> None:
        goated_stage = "models/goated_models"
        goated_dir = self._resolve_local_path(goated_stage)
        os.makedirs(goated_dir, exist_ok=True)
        model_name = best_info["model_name"]
        best_model_path = os.path.join(goated_dir, "best_model.pt")
        horizon_stage = best_info["stage"]
        horizon_dir = self._resolve_local_path(horizon_stage)
        src_model = os.path.join(horizon_dir, f"{model_name}.pt")
        try:
            shutil.copy2(src_model, best_model_path)
            self.logger.info(f"Best model copied to {best_model_path}")
        except Exception as e:
            self.logger.error(f"Failed to copy best model: {e}")
        if best_training_data_path is not None:
            best_training_data_dest = os.path.join(goated_dir, "best_model_training_data.csv")
            try:
                shutil.copy2(best_training_data_path, best_training_data_dest)
                self.logger.info(f"Best training data copied to {best_training_data_dest}")
            except Exception as e:
                self.logger.error(f"Failed to copy best training data: {e}")

    def _save_summary(self, results: list) -> None:
        from .model_analysis import ModelAnalysis
        analysis = ModelAnalysis(self.data_handler, model_stage="models")
        analysis.save_summary_table(results)
        self.logger.info("Training summary table saved.")
