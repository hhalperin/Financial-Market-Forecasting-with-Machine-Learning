"""
Model Pipeline Module

Coordinates data preparation and training across multiple time horizon combinations.
Enhancements include:
  - Sentiment filtering.
  - Outlier removal using IQR and z-score filtering.
  - Future horizon column removal.
  - Feature scaling.
  - Candidate architecture search.
  - Saving candidate models and summaries.
  - Tracking the global best ("goated") model based on various metrics.
  - Before training, it checks the temp folder (data/models/temp) for candidate model folders.
    If any exist, it finds the latest candidate (by modification time), looks it up in the horizons CSV 
    (data/models/horizons/horizons.csv) and resumes training from the next horizon combo.
"""

import os
import shutil
import datetime
import gc
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from typing import Any, Optional, List, Dict
from src.config import settings
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from src.utils.logger import get_logger
from .model_summary import ModelSummary

class ModelPipeline:
    def __init__(self, model_manager: Any, data_handler: Any, horizon_manager: Any) -> None:
        self.model_manager = model_manager
        self.data_handler = data_handler
        self.horizon_manager = horizon_manager
        self.logger = get_logger(self.__class__.__name__)
        # Create a summary manager to handle all "goated" model tracking & summary logic.
        self.summary_manager = ModelSummary(data_handler=self.data_handler)

    def advanced_filter_sentiment(self, df: pd.DataFrame, sentiment_columns: list = None,
                                  threshold: float = 0.2, mode: str = "any") -> pd.DataFrame:
        if not sentiment_columns:
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
        combined_mask = bool_masks[0]
        for m in bool_masks[1:]:
            if mode == "any":
                combined_mask = combined_mask | m
            else:
                combined_mask = combined_mask & m
        filtered_df = df[combined_mask].copy()
        self.logger.info(f"Advanced sentiment filter: reduced rows from {orig_len} to {len(filtered_df)} "
                         f"(mode='{mode}', threshold={threshold}, columns={valid_cols}).")
        return filtered_df

    def _drop_future_horizon_columns(self, X: pd.DataFrame, gather_horizon_str: str) -> pd.DataFrame:
        gather_mins = int(gather_horizon_str.split("_minutes")[0])
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
        return X[keep_cols]

    def _skip_training_if_small_change(self, df: pd.DataFrame, target_col: str, threshold: float = 0.01) -> bool:
        if target_col not in df.columns:
            return True
        avg_abs_change = df[target_col].abs().mean()
        if avg_abs_change < threshold:
            self.logger.info(f"Average target change {avg_abs_change:.4f} is below {threshold:.2%}. Skipping training.")
            return True
        return False

    def _resolve_local_path(self, stage_name: str) -> str:
        if self.data_handler.storage_mode == "s3":
            return os.path.join(self.data_handler.base_data_dir, stage_name)
        else:
            return os.path.join(self.data_handler.base_data_dir, stage_name)

    def _get_starting_index(self, combos: List[Dict[str, Any]]) -> int:
        temp_dir = os.path.join(self.data_handler.base_data_dir, "models", "temp")
        if not os.path.exists(temp_dir) or not os.listdir(temp_dir):
            self.logger.info("Temp directory is empty. Starting from the first horizon.")
            return 0
        candidate_names = [d for d in os.listdir(temp_dir) if d.startswith("model_") and "_to_" in d]
        if not candidate_names:
            return 0
        latest_candidate = max(candidate_names, key=lambda d: os.path.getmtime(os.path.join(temp_dir, d)))
        try:
            parts = latest_candidate.split("_to_")
            gather = parts[0].replace("model_", "")
            predict = parts[1]
        except Exception as e:
            self.logger.error(f"Error parsing candidate folder name {latest_candidate}: {e}")
            return 0
        horizons_csv_path = os.path.join(self.data_handler.base_data_dir, "models", "horizons", "horizons.csv")
        if not os.path.exists(horizons_csv_path):
            self.logger.warning(f"Horizons CSV not found at {horizons_csv_path}. Starting from index 0.")
            return 0
        horizons_df = pd.read_csv(horizons_csv_path)
        match = horizons_df[(horizons_df["gather_name"] == gather) & (horizons_df["predict_name"] == predict)]
        if match.empty:
            self.logger.warning(f"Candidate {latest_candidate} not found in horizons CSV. Starting from index 0.")
            return 0
        last_index = match.index[0]
        self.logger.info(f"Found latest candidate in horizons CSV at index {last_index}. Starting from next index.")
        return last_index + 1

    def _recursive_update_goated(self, metric_updates: List[tuple]) -> None:
        """Recursively update goated model metrics.
           Each tuple is (metric_name, global_best_info).
        """
        if not metric_updates:
            return
        metric_name, global_info = metric_updates[0]
        if global_info is not None:
            self.summary_manager.update_goated_model_metric(metric_name, global_info)
        self._recursive_update_goated(metric_updates[1:])

    def train_on_horizons(
        self,
        X: pd.DataFrame,
        df: pd.DataFrame,
        max_combos: int = 20000,
        save_best_only: bool = True,
        filter_sentiment: bool = False,
        sentiment_threshold: float = 0.35,
        sentiment_cols: list = None,
        sentiment_mode: str = "any",
        combos: Optional[List[Dict[str, Any]]] = None
    ) -> list:
        if filter_sentiment and sentiment_cols:
            df = self.advanced_filter_sentiment(df, sentiment_columns=sentiment_cols,
                                                threshold=sentiment_threshold, mode=sentiment_mode)
            X = X.loc[df.index]
        if combos is None:
            combos = self.horizon_manager.generate_horizon_combos()[:max_combos]
        start_index = self._get_starting_index(combos)
        if start_index > 0:
            self.logger.info(f"Resuming training from horizon combo index {start_index}.")
            combos = combos[start_index:]
        else:
            self.logger.info("Starting training from the first horizon combo.")

        results = []
        for idx, combo in enumerate(tqdm(combos, desc="Training across horizons", unit="horizon")):
            gather = combo["gather_name"]
            predict = combo["predict_name"]
            target_col = f"{predict}_percentage_change"
            if target_col not in df.columns:
                continue

            df_f = df.dropna(subset=[target_col])
            if df_f.empty:
                continue

            q1 = df_f[target_col].quantile(0.25)
            q3 = df_f[target_col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_f = df_f[(df_f[target_col] >= lower_bound) & (df_f[target_col] <= upper_bound)]
            if df_f.empty:
                continue

            from scipy.stats import zscore
            z_scores = zscore(df_f[target_col])
            df_f = df_f[(abs(z_scores) < 3.0)]
            if df_f.empty:
                self.logger.warning(
                    f"All data removed (z-score filter) for target {target_col}. Skipping horizon {gather} to {predict}."
                )
                continue

            X_f = X.loc[df_f.index]
            y = df_f[target_col].values
            if X_f.empty:
                continue

            X_f = self._drop_future_horizon_columns(X_f, gather)
            if self._skip_training_if_small_change(df_f, target_col, threshold=0.01):
                continue

            X_f_numeric = X_f.select_dtypes(include=["number"])
            if X_f_numeric.shape[1] == 0:
                self.logger.warning(f"No numeric features remain for horizon '{gather}' -> '{predict}'. Skipping.")
                continue

            if settings.use_scaler:
                if settings.scaler_type == "robust":
                    scaler = RobustScaler()
                elif settings.scaler_type == "standard":
                    scaler = StandardScaler()
                elif settings.scaler_type == "minmax":
                    scaler = MinMaxScaler()
                else:
                    scaler = None
                if scaler is not None:
                    X_f_numeric = pd.DataFrame(
                        scaler.fit_transform(X_f_numeric),
                        index=X_f_numeric.index,
                        columns=X_f_numeric.columns
                    )

            self.logger.info(f"Training with {X_f_numeric.shape[1]} numeric features: {list(X_f_numeric.columns)}. Target Column: {target_col}")
            candidate_model_name = f"model_{gather}_to_{predict}"
            self.logger.info(f"Training model: {candidate_model_name}")

            dest_folder_relative = os.path.join("models", "best_models", predict, candidate_model_name)
            dest_dir = self._resolve_local_path(dest_folder_relative)
            if os.path.exists(dest_dir):
                try:
                    shutil.rmtree(dest_dir)
                    self.logger.info(f"Cleared existing folder at {dest_dir}")
                except Exception as e:
                    self.logger.error(f"Failed to clear folder: {e}")
            os.makedirs(dest_dir, exist_ok=True)

            old_stage = self.model_manager.model_stage
            temp_candidate_folder = os.path.join(old_stage, "temp", candidate_model_name)
            self.model_manager.model_stage = temp_candidate_folder
            old_input_size = self.model_manager.input_size
            self.model_manager.input_size = X_f_numeric.shape[1]

            candidate_architectures = settings.candidate_architectures
            candidate_summaries: List[Dict[str, Any]] = []
            best_candidate_model = None
            best_candidate_metrics = None
            best_architecture = None
            original_hidden_layers = self.model_manager.hidden_layers

            for arch in candidate_architectures:
                self.logger.info(f"Trying architecture {arch} for model {candidate_model_name}")
                self.model_manager.hidden_layers = arch
                candidate_model, candidate_metrics = self.model_manager.train_and_evaluate(
                    X_f_numeric.values,
                    y,
                    candidate_model_name + f"_arch_{'_'.join(map(str, arch))}"
                )
                if candidate_model is None:
                    continue

                candidate_summary = {
                    "architecture": arch,
                    "metrics": {
                        "mse": candidate_metrics.get("mse"),
                        "mae": candidate_metrics.get("mae"),
                        "r2": candidate_metrics.get("r2"),
                        "explained_variance": candidate_metrics.get("explained_variance"),
                        "mape": candidate_metrics.get("mape"),
                        "regression_accuracy": candidate_metrics.get("regression_accuracy"),
                        "line_of_best_fit_error": candidate_metrics.get("line_of_best_fit_error"),
                        "directional_accuracy": candidate_metrics.get("directional_accuracy"),
                        "percentage_over_prediction": candidate_metrics.get("percentage_over_prediction"),
                        "pearson_correlation": candidate_metrics.get("pearson_correlation"),
                        "spearman_correlation": candidate_metrics.get("spearman_correlation")
                    }
                }
                candidate_summaries.append(candidate_summary)
                if best_candidate_metrics is None or candidate_metrics["line_of_best_fit_error"] < best_candidate_metrics["line_of_best_fit_error"]:
                    best_candidate_model = candidate_model
                    best_candidate_metrics = candidate_metrics
                    best_architecture = arch

            self.model_manager.hidden_layers = original_hidden_layers
            self.model_manager.input_size = old_input_size

            if best_candidate_model is None:
                self.logger.warning(f"No valid model trained for horizon {gather} to {predict}. Skipping.")
                self.model_manager.model_stage = old_stage
                continue

            model = best_candidate_model
            metrics = best_candidate_metrics
            self.logger.info(f"Best architecture for {candidate_model_name} is {best_architecture} with Line Fit Error {metrics['line_of_best_fit_error']:.4f}")

            if save_best_only:
                model_dest_path = os.path.join(dest_dir, "best_model.pt")
                self.model_manager.save_model(model, model_dest_path)
                train_df = X_f_numeric.copy()
                train_df[target_col] = y
                training_data_filename = "best_model_training_data.csv"
                self.data_handler.save_dataframe(train_df, training_data_filename, stage=dest_folder_relative)
                self.logger.info(f"Saved training data to {training_data_filename} in stage {dest_folder_relative}.")

                temp_candidate_dir = self._resolve_local_path(temp_candidate_folder)
                if os.path.exists(temp_candidate_dir):
                    for file in os.listdir(temp_candidate_dir):
                        if file.lower().endswith(".png") or file.lower().endswith(".json"):
                            src_file = os.path.join(temp_candidate_dir, file)
                            dest_file = os.path.join(dest_dir, file)
                            try:
                                shutil.copy2(src_file, dest_file)
                            except Exception as e:
                                self.logger.error(f"Failed to copy plot {file}: {e}")

                best_candidate_summary = {
                    "architecture": best_architecture,
                    "metrics": {
                        "mse": metrics.get("mse"),
                        "mae": metrics.get("mae"),
                        "r2": metrics.get("r2"),
                        "explained_variance": metrics.get("explained_variance"),
                        "mape": metrics.get("mape"),
                        "regression_accuracy": metrics.get("regression_accuracy"),
                        "line_of_best_fit_error": metrics.get("line_of_best_fit_error"),
                        "directional_accuracy": metrics.get("directional_accuracy"),
                        "percentage_over_prediction": metrics.get("percentage_over_prediction"),
                        "pearson_correlation": metrics.get("pearson_correlation"),
                        "spearman_correlation": metrics.get("spearman_correlation")
                    }
                }

                self.summary_manager.save_individual_model_summary(
                    dest_dir,
                    candidate_model_name,
                    gather,
                    predict,
                    candidate_summaries,
                    best_candidate_summary
                )

                # Update global best models for each metric.
                # (For each metric, update if the new candidate beats the global best.)
                if metrics["r2"] > self.summary_manager.global_best_r2:
                    self.summary_manager.global_best_r2 = metrics["r2"]
                    self.summary_manager.global_best_info_r2 = {
                        "model_name": candidate_model_name,
                        "dest_stage": dest_folder_relative,
                        "gather": gather,
                        "predict": predict,
                        "r2": metrics["r2"],
                        "architecture": best_architecture
                    }
                    self.summary_manager.update_goated_model_metric("r2", self.summary_manager.global_best_info_r2)

                if metrics["line_of_best_fit_error"] < self.summary_manager.global_best_lobf:
                    self.summary_manager.global_best_lobf = metrics["line_of_best_fit_error"]
                    self.summary_manager.global_best_info_lobf = {
                        "model_name": candidate_model_name,
                        "dest_stage": dest_folder_relative,
                        "gather": gather,
                        "predict": predict,
                        "lobf": metrics["line_of_best_fit_error"],
                        "architecture": best_architecture
                    }
                    self.summary_manager.update_goated_model_metric("lobf", self.summary_manager.global_best_info_lobf)

                if metrics.get("explained_variance") is not None and metrics["explained_variance"] > self.summary_manager.global_best_explained_variance:
                    self.summary_manager.global_best_explained_variance = metrics["explained_variance"]
                    self.summary_manager.global_best_info_explained_variance = {
                        "model_name": candidate_model_name,
                        "dest_stage": dest_folder_relative,
                        "gather": gather,
                        "predict": predict,
                        "explained_variance": metrics["explained_variance"],
                        "architecture": best_architecture
                    }
                    self.summary_manager.update_goated_model_metric("explained_variance", self.summary_manager.global_best_info_explained_variance)

                if metrics.get("mape") is not None and metrics["mape"] < self.summary_manager.global_best_mape:
                    self.summary_manager.global_best_mape = metrics["mape"]
                    self.summary_manager.global_best_info_mape = {
                        "model_name": candidate_model_name,
                        "dest_stage": dest_folder_relative,
                        "gather": gather,
                        "predict": predict,
                        "mape": metrics["mape"],
                        "architecture": best_architecture
                    }
                    self.summary_manager.update_goated_model_metric("mape", self.summary_manager.global_best_info_mape)

                if metrics.get("regression_accuracy") is not None and metrics["regression_accuracy"] > self.summary_manager.global_best_regression_accuracy:
                    self.summary_manager.global_best_regression_accuracy = metrics["regression_accuracy"]
                    self.summary_manager.global_best_info_regression_accuracy = {
                        "model_name": candidate_model_name,
                        "dest_stage": dest_folder_relative,
                        "gather": gather,
                        "predict": predict,
                        "regression_accuracy": metrics["regression_accuracy"],
                        "architecture": best_architecture
                    }
                    self.summary_manager.update_goated_model_metric("regression_accuracy", self.summary_manager.global_best_info_regression_accuracy)

                if metrics.get("directional_accuracy") is not None and metrics["directional_accuracy"] > self.summary_manager.global_best_directional_accuracy:
                    self.summary_manager.global_best_directional_accuracy = metrics["directional_accuracy"]
                    self.summary_manager.global_best_info_directional_accuracy = {
                        "model_name": candidate_model_name,
                        "dest_stage": dest_folder_relative,
                        "gather": gather,
                        "predict": predict,
                        "directional_accuracy": metrics["directional_accuracy"],
                        "architecture": best_architecture
                    }
                    self.summary_manager.update_goated_model_metric("directional_accuracy", self.summary_manager.global_best_info_directional_accuracy)

                if metrics.get("percentage_over_prediction") is not None and metrics["percentage_over_prediction"] < self.summary_manager.global_best_percentage_over_prediction:
                    self.summary_manager.global_best_percentage_over_prediction = metrics["percentage_over_prediction"]
                    self.summary_manager.global_best_info_percentage_over_prediction = {
                        "model_name": candidate_model_name,
                        "dest_stage": dest_folder_relative,
                        "gather": gather,
                        "predict": predict,
                        "percentage_over_prediction": metrics["percentage_over_prediction"],
                        "architecture": best_architecture
                    }
                    self.summary_manager.update_goated_model_metric("percentage_over_prediction", self.summary_manager.global_best_info_percentage_over_prediction)

                if metrics.get("pearson_correlation") is not None and metrics["pearson_correlation"] > self.summary_manager.global_best_pearson:
                    self.summary_manager.global_best_pearson = metrics["pearson_correlation"]
                    self.summary_manager.global_best_info_pearson = {
                        "model_name": candidate_model_name,
                        "dest_stage": dest_folder_relative,
                        "gather": gather,
                        "predict": predict,
                        "pearson_correlation": metrics["pearson_correlation"],
                        "architecture": best_architecture
                    }
                    self.summary_manager.update_goated_model_metric("pearson_correlation", self.summary_manager.global_best_info_pearson)

                if metrics.get("spearman_correlation") is not None and metrics["spearman_correlation"] > self.summary_manager.global_best_spearman:
                    self.summary_manager.global_best_spearman = metrics["spearman_correlation"]
                    self.summary_manager.global_best_info_spearman = {
                        "model_name": candidate_model_name,
                        "dest_stage": dest_folder_relative,
                        "gather": gather,
                        "predict": predict,
                        "spearman_correlation": metrics["spearman_correlation"],
                        "architecture": best_architecture
                    }
                    self.summary_manager.update_goated_model_metric("spearman_correlation", self.summary_manager.global_best_info_spearman)

            results.append({
                "gather_horizon": gather,
                "predict_horizon": predict,
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "architecture": best_architecture,
                "line_of_best_fit_error": metrics["line_of_best_fit_error"]
            })

            self.model_manager.model_stage = old_stage

            if (idx + 1) % 1000 == 0:
                self.summary_manager.clear_memory()
                self.logger.info(f"Memory cleared after processing {idx + 1} models.")

        if not results:
            self.logger.warning("No horizon combinations produced any results.")
            return results

        self.summary_manager.save_summary(results)
        self._recursive_update_goated([
            ("r2", self.summary_manager.global_best_info_r2),
            ("lobf", self.summary_manager.global_best_info_lobf),
            ("explained_variance", self.summary_manager.global_best_info_explained_variance),
            ("mape", self.summary_manager.global_best_info_mape),
            ("regression_accuracy", self.summary_manager.global_best_info_regression_accuracy),
            ("directional_accuracy", self.summary_manager.global_best_info_directional_accuracy),
            ("percentage_over_prediction", self.summary_manager.global_best_info_percentage_over_prediction),
            ("pearson_correlation", self.summary_manager.global_best_info_pearson),
            ("spearman_correlation", self.summary_manager.global_best_info_spearman)
        ])
        return results
