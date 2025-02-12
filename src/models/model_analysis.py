"""
Model Analysis Module

This module handles model evaluation and analysis. It computes various metrics (MSE, MAE, R², etc.),
generates plots (such as learning curves, actual vs. predicted plots, and SHAP feature importance), and saves
summaries and model details. It also provides multiple visual analyses of the model's performance.
"""

import os
import io
import json
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from src.utils.logger import get_logger
import torch
from typing import Any, Optional
from scipy import stats


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
        """
        Plots the learning curve (train and validation loss vs. epochs).
        """
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
        """
        Plots the actual versus predicted values.
        """
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

    # def plot_shap_feature_importance(self, model: torch.nn.Module, X_np: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
    #     """
    #     Generates a SHAP summary (beeswarm) plot.
    #     """
    #     figure_filename = f"{model_name}_shap_importance.png"
    #     if value is not None:
    #         figure_filename = f"{model_name}_shap_importance_{value:.4f}.png"
    #     X_torch = torch.tensor(X_np, dtype=torch.float32)
    #     model.eval()
    #     background = X_torch[:100]
    #     try:
    #         explainer = shap.DeepExplainer(model, background)
    #         shap_values = explainer.shap_values(X_torch)
    #         shap.summary_plot(shap_values, X_np, show=False)
    #         self._save_figure(figure_filename)
    #     except Exception as e:
    #         self.logger.error(f"SHAP analysis failed: {e}")

    # def plot_shap_importance_bar(self, model: torch.nn.Module, X_np: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
    #     """
    #     Generates a bar chart of the mean absolute SHAP values for each feature.
    #     Fix: Uses the shape of the computed SHAP values.
    #     """
    #     figure_filename = f"{model_name}_shap_importance_bar.png"
    #     if value is not None:
    #         figure_filename = f"{model_name}_shap_importance_bar_{value:.4f}.png"
    #     X_torch = torch.tensor(X_np, dtype=torch.float32)
    #     model.eval()
    #     background = X_torch[:100]
    #     try:
    #         explainer = shap.DeepExplainer(model, background)
    #         shap_values = explainer.shap_values(X_torch)[0]  # For regression, use first output.
    #         mean_abs_shap = np.abs(shap_values).mean(axis=0)
    #         # Use the actual length from the SHAP values.
    #         num_features = mean_abs_shap.shape[0]
    #         feature_names = [f"Feature_{i}" for i in range(num_features)]
    #         df_shap = pd.DataFrame({
    #             "feature": feature_names,
    #             "mean_abs_shap": mean_abs_shap
    #         }).sort_values("mean_abs_shap", ascending=True)
    #         plt.figure(figsize=(12, 8))
    #         plt.barh(df_shap["feature"], df_shap["mean_abs_shap"])
    #         plt.xlabel("Mean Absolute SHAP Value")
    #         plt.title("Feature Importance (SHAP)")
    #         plt.tight_layout()
    #         self._save_figure(figure_filename)
    #     except Exception as e:
    #         self.logger.error(f"SHAP bar plot failed: {e}")

    # --- New Analysis Methods ---

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
        """
        Plots residuals (errors) vs. predicted values.
        """
        figure_filename = f"{model_name}_residuals.png"
        if value is not None:
            figure_filename = f"{model_name}_residuals_{value:.4f}.png"
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
        """
        Plots a histogram of the residual errors.
        """
        figure_filename = f"{model_name}_error_histogram.png"
        if value is not None:
            figure_filename = f"{model_name}_error_histogram_{value:.4f}.png"
        errors = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.title("Histogram of Residual Errors")
        plt.grid(True)
        self._save_figure(figure_filename)

    def plot_qq(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
        """
        Generates a Q-Q plot of the residuals.
        """
        figure_filename = f"{model_name}_qq_plot.png"
        if value is not None:
            figure_filename = f"{model_name}_qq_plot_{value:.4f}.png"
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title("Q-Q Plot of Residuals")
        self._save_figure(figure_filename)

    def plot_error_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
        """
        Plots absolute errors versus actual values.
        """
        figure_filename = f"{model_name}_error_vs_actual.png"
        if value is not None:
            figure_filename = f"{model_name}_error_vs_actual_{value:.4f}.png"
        errors = np.abs(y_true - y_pred)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, errors, alpha=0.6, color='purple')
        plt.xlabel("Actual Values")
        plt.ylabel("Absolute Error")
        plt.title("Absolute Error vs. Actual Values")
        plt.grid(True)
        self._save_figure(figure_filename)

    def plot_feature_vs_prediction(self, X_df: pd.DataFrame, y_pred: np.ndarray, feature_name: str, model_name: str) -> None:
        """
        Plots one feature against the model's predicted output.
        """
        figure_filename = f"{model_name}_feature_vs_prediction_{feature_name}.png"
        plt.figure(figsize=(10, 6))
        plt.scatter(X_df[feature_name], y_pred, alpha=0.6)
        plt.xlabel(feature_name)
        plt.ylabel("Predicted Value")
        plt.title(f"{feature_name} vs. Predicted Value")
        plt.grid(True)
        self._save_figure(figure_filename)

    def plot_correlation_heatmap(self, X_df: pd.DataFrame, y: pd.Series, model_name: str) -> None:
        """
        Plots a heatmap of the correlation matrix among features and between features and the target.
        """
        figure_filename = f"{model_name}_correlation_heatmap.png"
        # Concatenate y as a column in X_df for correlation
        df_corr = X_df.copy()
        df_corr['target'] = y
        corr_matrix = df_corr.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        self._save_figure(figure_filename)

    def plot_boxplot_errors(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
        """
        Creates a boxplot of the residual errors.
        """
        figure_filename = f"{model_name}_boxplot_errors.png"
        if value is not None:
            figure_filename = f"{model_name}_boxplot_errors_{value:.4f}.png"
        errors = y_true - y_pred
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=errors)
        plt.title("Boxplot of Residual Errors")
        plt.ylabel("Error")
        self._save_figure(figure_filename)

    def plot_hexbin_scatter(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
        """
        Creates a hexbin scatter plot of actual vs. predicted values.
        """
        figure_filename = f"{model_name}_hexbin_scatter.png"
        if value is not None:
            figure_filename = f"{model_name}_hexbin_scatter_{value:.4f}.png"
        plt.figure(figsize=(10, 6))
        plt.hexbin(y_true, y_pred, gridsize=30, cmap='Blues', mincnt=1)
        plt.colorbar(label='Count')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Hexbin Scatter Plot: Actual vs. Predicted")
        self._save_figure(figure_filename)

    def plot_learning_curve_with_error(self, training_history: dict, model_name: str, value: Optional[float] = None) -> None:
        """
        Plots the learning curve with simulated error bars computed as the rolling standard deviation of loss.
        """
        figure_filename = f"{model_name}_learning_curve_with_error.png"
        if value is not None:
            figure_filename = f"{model_name}_learning_curve_with_error_{value:.4f}.png"
        train_losses = np.array(training_history["train_loss"])
        val_losses = np.array(training_history["val_loss"])
        # Compute rolling standard deviation with window size 3 (as a proxy for error bars)
        window = 3
        def rolling_std(x):
            return np.array([np.std(x[max(0, i-window+1):i+1]) for i in range(len(x))])
        train_std = rolling_std(train_losses)
        val_std = rolling_std(val_losses)
        epochs = range(1, len(train_losses)+1)
        plt.figure(figsize=(10, 6))
        plt.errorbar(epochs, train_losses, yerr=train_std, label="Train Loss", fmt='-o', capsize=3)
        plt.errorbar(epochs, val_losses, yerr=val_std, label="Validation Loss", fmt='-o', capsize=3)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Learning Curve with Error Bars")
        plt.legend()
        plt.grid(True)
        self._save_figure(figure_filename)

    def generate_all_plots(self, training_history: dict, y_true: np.ndarray, y_pred: np.ndarray,
                           model: torch.nn.Module, X_np: np.ndarray, model_name: str, value: Optional[float] = None) -> None:
        """
        Generates all performance plots.
        """
        self.plot_learning_curve(training_history, model_name, value)
        self.plot_learning_curve_with_error(training_history, model_name, value)
        self.plot_actual_vs_predicted(y_true, y_pred, model_name, value)
        #self.plot_shap_feature_importance(model, X_np, model_name, value)
        # self.plot_shap_importance_bar(model, X_np, model_name, value)
        self.plot_residuals(y_true, y_pred, model_name, value)
        self.plot_error_histogram(y_true, y_pred, model_name, value)
        self.plot_qq(y_true, y_pred, model_name, value)
        self.plot_error_vs_actual(y_true, y_pred, model_name, value)
        # If X_df is available with column names, you can call plot_feature_vs_prediction for selected features.
        # For demonstration, we assume X_np is converted to DataFrame with generic column names.
        # Here, we plot for the first feature.
        try:
            # Attempt to create a DataFrame using the number of columns in X_np.
            df_features = pd.DataFrame(X_np, columns=[f"Feature_{i}" for i in range(X_np.shape[1])])
            self.plot_feature_vs_prediction(df_features, y_pred, "Feature_0", model_name)
        except Exception as e:
            self.logger.error(f"Plotting feature vs. prediction failed: {e}")
        self.plot_correlation_heatmap(pd.DataFrame(X_np), y_true, model_name)
        self.plot_boxplot_errors(y_true, y_pred, model_name, value)
        self.plot_hexbin_scatter(y_true, y_pred, model_name, value)

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

    def save_model_summary(self, ticker: str, date_range: str, time_horizon: str,
                           metrics: dict, parameters: dict, X_df: pd.DataFrame, y: pd.Series) -> None:
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
