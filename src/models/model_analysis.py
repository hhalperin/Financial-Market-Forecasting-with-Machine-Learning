# src/models/model_analysis.py

import os
import io
import shap
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.logger import get_logger
import torch
import seaborn as sns  # <--- Make sure "seaborn" is in your requirements.txt

class ModelAnalysis:
    """
    Handles model evaluation, producing plots, and saving metrics.
    """

    def __init__(self, data_handler, model_stage="models", model_name="model"):
        self.data_handler = data_handler
        self.model_stage = model_stage
        self.model_name = model_name
        self.logger = get_logger(self.__class__.__name__)

    def compute_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        self.logger.info(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        return {"mse": mse, "mae": mae, "r2": r2}

    def plot_learning_curve(self, training_history, model_name, value=None):
        figure_filename = f"{model_name}_learning_curve.png"
        plt.figure(figsize=(10, 6))
        plt.plot(training_history["train_loss"], label="Train Loss")
        plt.plot(training_history["val_loss"], label="Val Loss")
        title_text = "Learning Curve"
        if value is not None:
            title_text += f" (Final MSE: {value:.4f})"
        plt.title(title_text)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        self._save_figure(figure_filename)

    def plot_actual_vs_predicted(self, y_true, y_pred, model_name, value=None):
        figure_filename = f"{model_name}_actual_vs_predicted"
        if value is not None:
            figure_filename += f"_{value:.4f}"
        figure_filename += ".png"

        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_true)), y_true, label="Actual", color="blue", alpha=0.6)
        plt.scatter(range(len(y_pred)), y_pred, label="Predicted", color="red", alpha=0.6)
        title_text = "Actual vs Predicted"
        if value is not None:
            title_text += f" (MSE: {value:.4f})"
        plt.title(title_text)
        plt.xlabel("Samples")
        plt.ylabel("Stock Price Change (%)")
        plt.legend()
        plt.grid(True)
        self._save_figure(figure_filename)

    def plot_shap_feature_importance(self, model, X_np, model_name, value=None):
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
            self.logger.error(f"SHAP failed: {e}")

    def plot_feature_correlation_heatmap(self, X_np, y_true, model_name):
        """
        Creates a correlation matrix among all features + 1 column for the target,
        then plots a seaborn heatmap of correlation.
        """
        figure_filename = f"{model_name}_correlation_heatmap.png"

        # Combine features + target
        # If you have column names for X_np, pass them in or do range-based.
        d = X_np.shape[1]
        feature_names = [f"feat_{i}" for i in range(d)]
        df_features = pd.DataFrame(X_np, columns=feature_names)
        df_features["target"] = y_true

        corr = df_features.corr(method="pearson")  # or "spearman"
        plt.figure(figsize=(max(8, d/2), max(8, d/2)))  # scale fig size by # of features maybe
        sns.heatmap(corr, annot=False, cmap="coolwarm", square=True)
        plt.title(f"Correlation Heatmap (Features + Target)")
        self._save_figure(figure_filename)

    def generate_all_plots(self, training_history, y_true, y_pred, model, X_np, model_name, value=None):
        self.plot_learning_curve(training_history, model_name, value)
        self.plot_actual_vs_predicted(y_true, y_pred, model_name, value)
        self.plot_shap_feature_importance(model, X_np, model_name, value)
        # Add the new correlation heatmap
        self.plot_feature_correlation_heatmap(X_np, y_true, model_name)

    def save_summary_table(self, results):
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

    def _save_model_details(self, results, filename):
        details = {"trials": results}
        self.data_handler.save_json(details, filename, stage=self.model_stage)

    def save_json_summary(self, trial_results):
        if not trial_results:
            self.logger.warning("No trial results to save.")
            return
        json_filename = f"{self.model_name}_trial_results.json"
        self.data_handler.save_json(trial_results, json_filename, stage=self.model_stage)
        self.logger.info(f"Trial results saved to JSON: {json_filename}")

    def _save_figure(self, figure_filename):
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        self.data_handler.save_figure_bytes(buffer.read(), figure_filename, stage=self.model_stage)
        plt.close()
