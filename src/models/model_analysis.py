import os
import io
import shap
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.logger import get_logger


class ModelAnalysis:
    """
    Handles model evaluation, producing plots and saving metrics.
    """

    def __init__(self, data_handler, model_stage="models", model_name="model"):
        self.data_handler = data_handler
        self.model_stage = model_stage
        self.model_name = model_name
        self.logger = get_logger(self.__class__.__name__)

    def compute_metrics(self, y_true, y_pred):
        """
        Returns dict of MSE, MAE, R^2 metrics.
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        self.logger.info(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        return {"mse": mse, "mae": mae, "r2": r2}

    def plot_learning_curve(self, training_history, model_name, value=None):
        """
        Plots train/val loss over epochs. Optionally includes final metric (e.g., MSE) in the title.
        """
        filename = f"{model_name}_learning_curve.png"
        plt.figure(figsize=(10, 6))
        plt.plot(training_history["train_loss"], label="Train Loss")
        plt.plot(training_history["val_loss"], label="Val Loss")
        title_text = f"Learning Curve"
        if value is not None:
            title_text += f" (Final MSE: {value:.4f})"
        plt.title(title_text)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        self._save_figure(filename)

    def plot_actual_vs_predicted(self, y_true, y_pred, model_name, value):
        """
        Scatters the actual vs. predicted for a quick visual check of model performance.
        """
        filename = f"{model_name}_actual_vs_predicted_{value:.4f}.png"
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_true)), y_true, label="Actual", color="blue", alpha=0.6)
        plt.scatter(range(len(y_pred)), y_pred, label="Predicted", color="red", alpha=0.6)
        plt.title(f"Actual vs Predicted (MSE: {value:.4f})")
        plt.xlabel("Samples")
        plt.ylabel("Stock Price Change (%)")
        plt.legend()
        plt.grid(True)
        self._save_figure(filename)

    def plot_shap_feature_importance(self, model, X_np, model_name, value):
        filename = f"{model_name}_shap_importance_{value:.4f}.png"

        import torch
        import shap

        X_torch = torch.tensor(X_np, dtype=torch.float32)

        # shap.DeepExplainer expects the model in evaluation mode
        model.eval()

        # Create a small sample of background data (for shap baseline)
        # e.g. the first 100 rows if your data is large
        background = X_torch[:100]

        # Initialize the DeepExplainer with your model + some background data
        explainer = shap.DeepExplainer(model, background)

        # Compute shap values on the entire dataset
        shap_values = explainer.shap_values(X_torch)

        # shap_values should be a NumPy array or list-of-arrays
        shap.summary_plot(shap_values, X_np, show=False)

        self._save_figure(filename)


    def generate_all_plots(self, training_history, y_true, y_pred, model, X, model_name, value):
        """
        Helper to produce all relevant plots after a training run.
        """
        self.plot_learning_curve(training_history, model_name, value)
        self.plot_actual_vs_predicted(y_true, y_pred, model_name, value)
        self.plot_shap_feature_importance(model, X, model_name, value)

    def save_summary_table(self, results):
        """
        Save a summary table of training results to a CSV file.
        :param results: List of dictionaries containing results.
        """
        if not results:
            self.logger.warning("No results to save in summary table.")
            return

        # Save results as a CSV
        df = pd.DataFrame(results)
        summary_filename = f"{self.model_name}_summary.csv"
        self.data_handler.save_dataframe(df, summary_filename, stage=self.model_stage)
        self.logger.info(f"Summary table saved: {summary_filename}")

        # Save detailed model information as JSON
        details_filename = f"{self.model_name}_details.json"
        self._save_model_details(results, details_filename)
        self.logger.info(f"Model details saved: {details_filename}")

    def _save_model_details(self, results, filename):
        """
        Save detailed model information to a JSON file.
        :param results: List of dictionaries containing model information.
        """
        details = {"trials": results}
        self.data_handler.save_json(details, filename, stage=self.model_stage)

    def save_json_summary(self, trial_results):
        """
        Save detailed trial results in JSON format.
        """
        if not trial_results:
            self.logger.warning("No trial results to save.")
            return

        json_filename = f"{self.model_name}_trial_results.json"
        self.data_handler.save_json(trial_results, json_filename, stage=self.model_stage)
        self.logger.info(f"Trial results saved to JSON: {json_filename}")

    def _save_figure(self, filename):
        """
        Save the current matplotlib figure to the data_handler, then close it.
        """
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        self.data_handler.save_figure_bytes(buffer.read(), filename, stage=self.model_stage)
        plt.close()
