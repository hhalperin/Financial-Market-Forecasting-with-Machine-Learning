"""
Model Analysis Module

This module handles model evaluation and analysis. It computes various metrics (MSE, MAE, R², etc.),
generates plots (such as learning curves, actual vs. predicted plots, and SHAP feature importance), and saves
summaries and model details.
"""

import os
import io
import json

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from src.utils.logger import get_logger
import torch
from typing import Any, Optional
from scipy import stats

import matplotlib
matplotlib.use('Agg')  # Set headless backend
import matplotlib.pyplot as plt


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
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        evs = explained_variance_score(y_true, y_pred)
        self.logger.info(f"Computed Metrics -> MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Explained Variance: {evs:.4f}")
        return {"mse": mse, "mae": mae, "r2": r2, "explained_variance": evs}

    def _get_figure_filename(self, model_name: str, plot_type: str, value: Optional[float] = None) -> str:
        filename = f"{model_name}_{plot_type}"
        if value is not None:
            filename += f"_{value:.4f}"
        return f"{filename}.png"

    def plot_learning_curve(self, training_history: dict, model_name: str, value: Optional[float] = None) -> None:
        figure_filename = self._get_figure_filename(model_name, "learning_curve", value)
        plt.figure(figsize=(10, 6))
        plt.plot(training_history["train_loss"], label="Train Loss")
        plt.plot(training_history["val_loss"], label="Validation Loss")
        title_text = "Learning Curve" + (f" (Final MSE: {value:.4f})" if value is not None else "")
        plt.title(title_text)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        self._save_figure(figure_filename)

    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
        figure_filename = self._get_figure_filename(model_name, "actual_vs_predicted", value)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, color="blue", alpha=0.6, label="Predicted vs Actual")
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal (y=x)")
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        best_fit_line = np.poly1d((slope, intercept))
        x_vals = np.linspace(min_val, max_val, 100)
        plt.plot(x_vals, best_fit_line(x_vals), color="green", linestyle="-", 
                 label=f"Best Fit (slope: {slope:.3f}, intercept: {intercept:.3f})")
        title_text = "Actual vs Predicted" + (f" (MSE: {value:.4f})" if value is not None else "")
        plt.title(title_text)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.legend()
        plt.grid(True)
        self._save_figure(figure_filename)

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
        figure_filename = self._get_figure_filename(model_name, "residuals", value)
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Residuals vs. Predicted")
        plt.grid(True)
        self._save_figure(figure_filename)

    def plot_error_histogram(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
        figure_filename = self._get_figure_filename(model_name, "error_histogram", value)
        errors = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.title("Histogram of Residual Errors")
        plt.grid(True)
        self._save_figure(figure_filename)

    def plot_qq(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
        figure_filename = self._get_figure_filename(model_name, "qq_plot", value)
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title("Q-Q Plot of Residuals")
        self._save_figure(figure_filename)

    def plot_error_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
        figure_filename = self._get_figure_filename(model_name, "error_vs_actual", value)
        errors = np.abs(y_true - y_pred)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, errors, alpha=0.6, color='purple')
        plt.xlabel("Actual Values")
        plt.ylabel("Absolute Error")
        plt.title("Absolute Error vs. Actual Values")
        plt.grid(True)
        self._save_figure(figure_filename)

    def plot_feature_vs_prediction(self, X_df: pd.DataFrame, y_pred: np.ndarray, feature_name: str, model_name: str) -> None:
        figure_filename = f"{model_name}_feature_vs_prediction_{feature_name}.png"
        plt.figure(figsize=(10, 6))
        plt.scatter(X_df[feature_name], y_pred, alpha=0.6)
        plt.xlabel(feature_name)
        plt.ylabel("Predicted Value")
        plt.title(f"{feature_name} vs. Predicted Value")
        plt.grid(True)
        self._save_figure(figure_filename)

    def plot_correlation_heatmap(self, X_df: pd.DataFrame, y: pd.Series, model_name: str) -> None:
        figure_filename = f"{model_name}_correlation_heatmap.png"
        df_corr = X_df.copy()
        df_corr['target'] = y
        corr_matrix = df_corr.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        self._save_figure(figure_filename)

    def plot_boxplot_errors(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
        figure_filename = self._get_figure_filename(model_name, "boxplot_errors", value)
        errors = y_true - y_pred
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=errors)
        plt.title("Boxplot of Residual Errors")
        plt.ylabel("Error")
        self._save_figure(figure_filename)

    def plot_hexbin_scatter(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
        figure_filename = self._get_figure_filename(model_name, "hexbin_scatter", value)
        plt.figure(figsize=(10, 6))
        plt.hexbin(y_true, y_pred, gridsize=30, cmap='Blues', mincnt=1)
        plt.colorbar(label='Count')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Hexbin Scatter Plot: Actual vs. Predicted")
        self._save_figure(figure_filename)

    def plot_learning_curve_with_error(self, training_history: dict, model_name: str, value: Optional[float] = None) -> None:
        figure_filename = self._get_figure_filename(model_name, "learning_curve_with_error", value)
        train_losses = np.array(training_history["train_loss"])
        val_losses = np.array(training_history["val_loss"])
        window = 3

        def rolling_std(x):
            return np.array([np.std(x[max(0, i - window + 1):i + 1]) for i in range(len(x))])

        train_std = rolling_std(train_losses)
        val_std = rolling_std(val_losses)
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.errorbar(epochs, train_losses, yerr=train_std, label="Train Loss", fmt='-o', capsize=3)
        plt.errorbar(epochs, val_losses, yerr=val_std, label="Validation Loss", fmt='-o', capsize=3)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Learning Curve with Error Bars")
        plt.legend()
        plt.grid(True)
        self._save_figure(figure_filename)

    def plot_target_histogram(self, y: np.ndarray, model_name: str) -> None:
        figure_filename = self._get_figure_filename(model_name, "target_histogram")
        plt.figure(figsize=(10, 6))
        plt.hist(y, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel("Target Value")
        plt.ylabel("Frequency")
        plt.title("Histogram of Target Values")
        plt.grid(True)
        self._save_figure(figure_filename)

    def generate_all_plots(self, training_history: dict, y_true: np.ndarray, y_pred: np.ndarray,
                           model: torch.nn.Module, X_np: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
        self.plot_learning_curve(training_history, model_name, value)
        self.plot_learning_curve_with_error(training_history, model_name, value)
        self.plot_actual_vs_predicted(y_true, y_pred, model_name, value)
        self.plot_residuals(y_true, y_pred, model_name, value)
        self.plot_error_histogram(y_true, y_pred, model_name, value)
        self.plot_qq(y_true, y_pred, model_name, value)
        self.plot_error_vs_actual(y_true, y_pred, model_name, value)
        self.plot_correlation_heatmap(pd.DataFrame(X_np), y_true, model_name)
        self.plot_boxplot_errors(y_true, y_pred, model_name, value)
        self.plot_hexbin_scatter(y_true, y_pred, model_name, value)
        self.plot_target_histogram(y_true, model_name)

    def save_summary_table(self, results: list) -> None:
        if not results:
            self.logger.warning("No results to save in summary table.")
            return
        df = pd.DataFrame(results)
        summary_filename = f"{self.model_name}_summary.csv"
        self.data_handler.save_dataframe(df, summary_filename, stage=self.model_stage)
        details_filename = f"{self.model_name}_details.json"
        self._save_model_details(results, details_filename)

    def _save_model_details(self, results: list, filename: str) -> None:
        details = {"trials": results}
        self.data_handler.save_json(details, filename, stage=self.model_stage)

    def save_json_summary(self, trial_results: Any) -> None:
        if not trial_results:
            self.logger.warning("No trial results to save.")
            return
        json_filename = f"{self.model_name}_trial_results.json"
        self.data_handler.save_json(trial_results, json_filename, stage=self.model_stage)

    def _save_figure(self, figure_filename: str) -> None:
        try:
            buffer = io.BytesIO()
            # Save the figure at a lower DPI to reduce memory usage.
            plt.savefig(buffer, format="png", dpi=100)
            buffer.seek(0)
            self.data_handler.save_figure_bytes(buffer.read(), figure_filename, stage=self.model_stage)
        except Exception as e:
            self.logger.error(f"Failed to save figure {figure_filename}: {e}")
        finally:
            plt.close('all')
            import gc
            gc.collect()



    def save_training_data(self, training_df: pd.DataFrame, filename: str) -> None:
        self.data_handler.save_dataframe(training_df, filename, stage=self.model_stage)

    def save_model_summary(self, ticker: str, date_range: str, time_horizon: str,
                           metrics: dict, parameters: dict, X_df: pd.DataFrame, y: pd.Series) -> None:
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