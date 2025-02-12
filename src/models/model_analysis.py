"""
Model Analysis Module

This module handles model evaluation and analysis. It computes various metrics (MSE, MAE, R², etc.), generates plots
(such as learning curves, actual vs. predicted plots, and SHAP feature importance), and saves summaries and model details.
It also identifies which features have the highest impact on the model's output.
"""

import os
import io
import shap
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from src.utils.logger import get_logger
import torch
import seaborn as sns  
from typing import Any, Optional

class ModelAnalysis:
    """
    Provides methods for analyzing a trained model.
    """
    def __init__(self, data_handler: Any, model_stage: str = "models", model_name: str = "model") -> None:
        self.data_handler = data_handler
        self.model_stage = model_stage
        self.model_name = model_name
        self.logger = get_logger(self.__class__.__name__)

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Computes evaluation metrics.
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        evs = explained_variance_score(y_true, y_pred)
        self.logger.info(f"Computed Metrics -> MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Explained Variance: {evs:.4f}")
        return {"mse": mse, "mae": mae, "r2": r2, "explained_variance": evs}

    def plot_learning_curve(self, training_history: dict, model_name: str, value: Optional[float] = None) -> None:
        figure_filename = f"{model_name}_learning_curve.png"
        plt.figure(figsize=(10, 6))
        plt.plot(training_history["train_loss"], label="Train Loss")
        plt.plot(training_history["val_loss"], label="Validation Loss")
        title_text = "Learning Curve"
        if value is not None:
            title_text += f" (Final MSE: {value:.4f})"
        plt.title(title_text)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        self._save_figure(figure_filename)

    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
        figure_filename = f"{model_name}_actual_vs_predicted"
        if value is not None:
            figure_filename += f"_{value:.4f}"
        figure_filename += ".png"
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, color="blue", alpha=0.6, label="Predicted vs Actual")
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal (y=x)")
        title_text = "Actual vs Predicted"
        if value is not None:
            title_text += f" (MSE: {value:.4f})"
        plt.title(title_text)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.legend()
        plt.grid(True)
        self._save_figure(figure_filename)

    def plot_shap_feature_importance(self, model: torch.nn.Module, X_np: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
        figure_filename = f"{model_name}_shap_importance"
        if value is not None:
            figure_filename += f"_{value:.4f}"
        figure_filename += ".png"
        X_torch = torch.tensor(X_np, dtype=torch.float32)
        model.eval()
        background = X_torch[:100]
        try:
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(X_torch)
            shap.summary_plot(shap_values, X_np, show=False)
            self._save_figure(figure_filename)
        except Exception as e:
            self.logger.error(f"SHAP analysis failed: {e}")

    def generate_all_plots(self, training_history: dict, y_true: np.ndarray, y_pred: np.ndarray, model: torch.nn.Module, X_np: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
        self.plot_learning_curve(training_history, model_name, value)
        self.plot_actual_vs_predicted(y_true, y_pred, model_name, value)
        self.plot_shap_feature_importance(model, X_np, model_name, value)

    def save_summary_table(self, results: list) -> None:
        if not results:
            self.logger.warning("No results to save in summary table.")
            return
        df = pd.DataFrame(results)
        summary_filename = f"{self.model_name}_summary.csv"
        self.data_handler.save_dataframe(df, summary_filename, stage=self.model_stage)
        self.logger.info(f"Summary table saved: {summary_filename}")
        details_filename = f"{self.model_name}_details.json"
        self._save_model_details(results, details_filename)
        self.logger.info(f"Model details saved: {details_filename}")

    def _save_model_details(self, results: list, filename: str) -> None:
        details = {"trials": results}
        self.data_handler.save_json(details, filename, stage=self.model_stage)

    def save_json_summary(self, trial_results: Any) -> None:
        if not trial_results:
            self.logger.warning("No trial results to save.")
            return
        json_filename = f"{self.model_name}_trial_results.json"
        self.data_handler.save_json(trial_results, json_filename, stage=self.model_stage)
        self.logger.info(f"Trial results saved to JSON: {json_filename}")

    def _save_figure(self, figure_filename: str) -> None:
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        self.data_handler.save_figure_bytes(buffer.read(), figure_filename, stage=self.model_stage)
        plt.close()

    def save_training_data(self, training_df: pd.DataFrame, filename: str) -> None:
        self.data_handler.save_dataframe(training_df, filename, stage=self.model_stage)
        self.logger.info(f"Training data saved as {filename} in stage {self.model_stage}.")

    def compute_feature_correlations(self, X_df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Computes correlations between each feature in X_df and the target y.
        Returns a DataFrame sorted by the absolute correlation values.
        """
        corr_dict = {col: X_df[col].corr(y) for col in X_df.columns}
        corr_df = pd.DataFrame(list(corr_dict.items()), columns=["Feature", "Correlation"])
        corr_df["AbsCorrelation"] = corr_df["Correlation"].abs()
        corr_df.sort_values("AbsCorrelation", ascending=False, inplace=True)
        corr_filename = f"{self.model_name}_feature_correlations.csv"
        self.data_handler.save_dataframe(corr_df, corr_filename, stage=self.model_stage)
        self.logger.info(f"Feature correlations saved as {corr_filename}.")
        return corr_df

    def save_model_summary(self, ticker: str, date_range: str, time_horizon: str, metrics: dict, parameters: dict, X_df: pd.DataFrame, y: pd.Series) -> None:
        """
        Saves a summary of the model performance along with the top five most critical features.
        """
        corr_df = self.compute_feature_correlations(X_df, y)
        top_features = corr_df.head(5).to_dict(orient="records")
        
        summary = {
            "model": {
                "ticker": ticker,
                "date_range": date_range,
                "time_horizon": time_horizon,
                "summary_details": {
                    "mse": metrics.get("mse"),
                    "mae": metrics.get("mae"),
                    "r2": metrics.get("r2"),
                    "explained_variance": metrics.get("explained_variance"),
                    "critical_model_feature_columns": top_features
                },
                "parameters": parameters
            }
        }
        summary_filename = f"{self.model_name}_model_summary.json"
        self.data_handler.save_json(summary, summary_filename, stage=self.model_stage)
        self.logger.info(f"Model summary saved as {summary_filename}.")
