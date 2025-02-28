"""
Model Pipeline Module

Coordinates data preparation and training across multiple time horizon combinations.
"""

import os
import shutil
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
        """
        Initializes the ModelPipeline.
        
        :param model_manager: Instance of ModelManager.
        :param data_handler: Instance of DataHandler.
        :param horizon_manager: Instance of TimeHorizonManager.
        """
        self.model_manager = model_manager
        self.data_handler = data_handler
        self.horizon_manager = horizon_manager
        self.logger = get_logger(self.__class__.__name__)
        self.summary_manager = ModelSummary(data_handler=self.data_handler)

    def advanced_filter_sentiment(self, df: pd.DataFrame, sentiment_columns: List[str] = None,
                                  threshold: float = 0.2, mode: str = "any") -> pd.DataFrame:
        """
        Filters the DataFrame rows based on sentiment column thresholds.
        
        :param df: Input DataFrame.
        :param sentiment_columns: List of sentiment columns to filter on.
        :param threshold: Threshold value for sentiment.
        :param mode: "any" for union, "all" for intersection.
        :return: Filtered DataFrame.
        """
        if not sentiment_columns:
            return df
        valid_cols = [col for col in sentiment_columns if col in df.columns]
        if not valid_cols:
            self.logger.warning("No valid sentiment columns found. Returning DataFrame unchanged.")
            return df
        orig_len = len(df)
        bool_masks = [df[col] >= threshold for col in valid_cols]
        combined_mask = bool_masks[0]
        for mask in bool_masks[1:]:
            combined_mask = combined_mask | mask if mode == "any" else combined_mask & mask
        filtered_df = df[combined_mask].copy()
        self.logger.info(f"Sentiment filter: reduced rows from {orig_len} to {len(filtered_df)} (mode='{mode}', threshold={threshold}, columns={valid_cols}).")
        return filtered_df

    def _drop_future_horizon_columns(self, X: pd.DataFrame, gather_horizon_str: str) -> pd.DataFrame:
        """
        Retains only columns with minute values less than or equal to the gather horizon.
        
        :param X: Input DataFrame.
        :param gather_horizon_str: Gather horizon string (e.g. "5_minutes").
        :return: Filtered DataFrame.
        """
        try:
            gather_mins = int(gather_horizon_str.split("_minutes")[0])
        except (ValueError, IndexError):
            return X
        keep_cols = []
        for col in X.columns:
            if "_minutes" in col:
                try:
                    if int(col.split("_minutes")[0]) <= gather_mins:
                        keep_cols.append(col)
                except ValueError:
                    keep_cols.append(col)
            else:
                keep_cols.append(col)
        return X[keep_cols]

    def _skip_training_if_small_change(self, df: pd.DataFrame, target_col: str, threshold: float = 0.01) -> bool:
        """
        Checks if the average absolute change in the target column is below a threshold.
        
        :param df: Input DataFrame.
        :param target_col: Target column name.
        :param threshold: Threshold value.
        :return: True if training should be skipped; False otherwise.
        """
        if target_col not in df.columns:
            return True
        avg_abs_change = df[target_col].abs().mean()
        if avg_abs_change < threshold:
            self.logger.info(f"Avg target change {avg_abs_change:.4f} is below {threshold:.2%}. Skipping training.")
            return True
        return False

    def _resolve_local_path(self, stage_name: str) -> str:
        """
        Returns the full local path for a given stage.
        
        :param stage_name: Stage folder name.
        :return: Full path string.
        """
        return os.path.join(self.data_handler.base_data_dir, stage_name)

    def _recursive_update_goated(self, metric_updates: List[tuple]) -> None:
        """
        Recursively updates global best metrics.
        
        :param metric_updates: List of (metric, best_info) tuples.
        """
        if not metric_updates:
            return
        metric, global_info = metric_updates[0]
        if global_info is not None:
            self.summary_manager.update_goated_model_metric(metric, global_info)
        self._recursive_update_goated(metric_updates[1:])

    def _update_global_best_metrics(self, candidate_model_name: str, dest_folder_relative: str,
                                    gather: str, predict: str, best_architecture: Any,
                                    metrics: dict, best_candidate_summary: dict) -> List[tuple]:
        """
        Updates global best metrics if the candidate is better.
        
        :param candidate_model_name: Name of candidate model.
        :param dest_folder_relative: Relative path for saving.
        :param gather: Gather horizon string.
        :param predict: Prediction horizon string.
        :param best_architecture: Best architecture details.
        :param metrics: Dictionary of metrics.
        :param best_candidate_summary: Summary of the best candidate.
        :return: List of updates.
        """
        metrics_to_check = {
            "r2": {"better": "higher", "global_attr": "global_best_r2", "info_attr": "global_best_info_r2"},
            "line_of_best_fit_error": {"better": "lower", "global_attr": "global_best_lobf", "info_attr": "global_best_info_lobf"},
            "explained_variance": {"better": "higher", "global_attr": "global_best_explained_variance", "info_attr": "global_best_info_explained_variance"},
            "mape": {"better": "lower", "global_attr": "global_best_mape", "info_attr": "global_best_info_mape"},
            "regression_accuracy": {"better": "higher", "global_attr": "global_best_regression_accuracy", "info_attr": "global_best_info_regression_accuracy"},
            "directional_accuracy": {"better": "higher", "global_attr": "global_best_directional_accuracy", "info_attr": "global_best_info_directional_accuracy"},
            "percentage_over_prediction": {"better": "lower", "global_attr": "global_best_percentage_over_prediction", "info_attr": "global_best_info_percentage_over_prediction"},
            "pearson_correlation": {"better": "higher", "global_attr": "global_best_pearson", "info_attr": "global_best_info_pearson"},
            "spearman_correlation": {"better": "higher", "global_attr": "global_best_spearman", "info_attr": "global_best_info_spearman"}
        }
        updates = []
        for metric, config in metrics_to_check.items():
            candidate_value = metrics.get(metric)
            if candidate_value is None:
                continue
            current_global = getattr(self.summary_manager, config["global_attr"])
            if (config["better"] == "higher" and candidate_value > current_global) or \
               (config["better"] == "lower" and candidate_value < current_global):
                setattr(self.summary_manager, config["global_attr"], candidate_value)
                best_info = {
                    "model_name": candidate_model_name,
                    "dest_stage": dest_folder_relative,
                    "gather": gather,
                    "predict": predict,
                    "architecture": best_architecture,
                    "metrics": best_candidate_summary["metrics"]
                }
                setattr(self.summary_manager, config["info_attr"], best_info)
                updates.append((metric, best_info))
        return updates

    def train_on_horizons(
        self,
        X: pd.DataFrame,
        df: pd.DataFrame,
        max_combos: int = 20000,
        save_best_only: bool = True,
        filter_sentiment: bool = False,
        sentiment_threshold: float = 0.35,
        sentiment_cols: List[str] = None,
        sentiment_mode: str = "any",
        combos: Optional[List[Dict[str, Any]]] = None
    ) -> list:
        """
        Trains models across different time horizon combinations.
        
        :param X: Feature DataFrame.
        :param df: Raw DataFrame (for target columns).
        :param max_combos: Maximum number of combinations to try.
        :param save_best_only: Whether to save only the best candidate.
        :param filter_sentiment: Whether to filter based on sentiment.
        :param sentiment_threshold: Threshold for sentiment filtering.
        :param sentiment_cols: List of sentiment column names.
        :param sentiment_mode: "any" or "all" filtering mode.
        :param combos: List of horizon combination dictionaries.
        :return: List of training result dictionaries.
        """
        if filter_sentiment and sentiment_cols:
            df = self.advanced_filter_sentiment(df, sentiment_columns=sentiment_cols,
                                                threshold=sentiment_threshold, mode=sentiment_mode)
            X = X.loc[df.index]
        if combos is None:
            combos = self.horizon_manager.generate_horizon_combos()[:max_combos]
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
            df_f = df_f[(abs(zscore(df_f[target_col])) < 3.0)]
            if df_f.empty:
                self.logger.warning(f"All data removed (z-score filter) for target {target_col}. Skipping {gather} to {predict}.")
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
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                elif settings.scaler_type == "standard":
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                elif settings.scaler_type == "minmax":
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                else:
                    scaler = None
                if scaler is not None:
                    X_f_numeric = pd.DataFrame(
                        scaler.fit_transform(X_f_numeric),
                        index=X_f_numeric.index,
                        columns=X_f_numeric.columns
                    )

            candidate_model_name = f"model_{gather}_to_{predict}"
            self.logger.info(f"Training model: {candidate_model_name}")

            if settings.save_only_goated:
                dest_folder_relative = os.path.join("models", "goated_models", predict, candidate_model_name)
            else:
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
            self.model_manager.model_stage = dest_folder_relative
            old_input_size = self.model_manager.input_size
            self.model_manager.input_size = X_f_numeric.shape[1]

            candidate_architectures = settings.candidate_architectures
            candidate_summaries = []
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
                    model_name=candidate_model_name + f"_arch_{'_'.join(map(str, arch))}"
                )
                if candidate_model is None:
                    continue

                candidate_summary = {
                    "architecture": arch,
                    "metrics": candidate_metrics
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

                from .model_summary import ModelSummary
                best_candidate_summary = {
                    "architecture": best_architecture,
                    "metrics": candidate_metrics
                }
                self.summary_manager.save_individual_model_summary(
                    dest_dir,
                    candidate_model_name,
                    gather,
                    predict,
                    candidate_summaries,
                    best_candidate_summary
                )

                metric_updates = self._update_global_best_metrics(
                    candidate_model_name, dest_folder_relative, gather, predict, best_architecture, metrics, best_candidate_summary
                )
                self._recursive_update_goated(metric_updates)

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
            ("line_of_best_fit_error", self.summary_manager.global_best_info_lobf),
            ("explained_variance", self.summary_manager.global_best_info_explained_variance),
            ("mape", self.summary_manager.global_best_info_mape),
            ("regression_accuracy", self.summary_manager.global_best_info_regression_accuracy),
            ("directional_accuracy", self.summary_manager.global_best_info_directional_accuracy),
            ("percentage_over_prediction", self.summary_manager.global_best_info_percentage_over_prediction),
            ("pearson_correlation", self.summary_manager.global_best_info_pearson),
            ("spearman_correlation", self.summary_manager.global_best_info_spearman)
        ])
        return results
