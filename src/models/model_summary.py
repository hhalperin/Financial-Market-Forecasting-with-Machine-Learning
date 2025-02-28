import os
import shutil
import datetime
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Dict
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_score, recall_score, f1_score
from src.utils.logger import get_logger
from .model_analysis import ModelAnalysis
from src.config import settings

class ModelSummary:
    def __init__(self, data_handler):
        """
        Manages model summary logic, tracks global best models, and computes additional metrics.
        
        :param data_handler: DataHandler instance.
        """
        self.data_handler = data_handler
        self.logger = get_logger(self.__class__.__name__)

        self.global_best_mse = float("inf")
        self.global_best_info_mse = None
        self.global_best_mae = float("inf")
        self.global_best_info_mae = None
        self.global_best_r2 = -float("inf")
        self.global_best_info_r2 = None
        self.global_best_lobf = float("inf")
        self.global_best_info_lobf = None

        self.global_best_pearson = -float("inf")
        self.global_best_info_pearson = None
        self.global_best_spearman = -float("inf")
        self.global_best_info_spearman = None
        self.global_best_precision = -float("inf")
        self.global_best_info_precision = None
        self.global_best_recall = -float("inf")
        self.global_best_info_recall = None
        self.global_best_f1 = -float("inf")
        self.global_best_info_f1 = None
        self.global_best_median_ae = float("inf")
        self.global_best_info_median_ae = None
        self.global_best_median_ape = float("inf")
        self.global_best_info_median_ape = None
        self.global_best_sharpe = -float("inf")
        self.global_best_info_sharpe = None

        self.global_best_explained_variance = -float("inf")
        self.global_best_info_explained_variance = None
        self.global_best_mape = float("inf")
        self.global_best_info_mape = None
        self.global_best_regression_accuracy = -float("inf")
        self.global_best_info_regression_accuracy = None
        self.global_best_directional_accuracy = -float("inf")
        self.global_best_info_directional_accuracy = None
        self.global_best_percentage_over_prediction = float("inf")
        self.global_best_info_percentage_over_prediction = None

    def save_individual_model_summary(
        self,
        candidate_model_name: str,
        gather: str,
        predict: str,
        candidate_summaries: List[Dict[str, Any]],
        best_candidate: Dict[str, Any]
    ) -> None:
        """
        Saves a JSON summary for a given prediction horizon.
        
        :param candidate_model_name: Candidate model name.
        :param gather: Gather horizon string.
        :param predict: Prediction horizon string.
        :param candidate_summaries: List of candidate summaries.
        :param best_candidate: Best candidate summary.
        """
        base_goated_dir = os.path.join(self.data_handler.base_data_dir, "models", "goated_models")
        os.makedirs(base_goated_dir, exist_ok=True)
        predict_folder = predict
        target_dir = os.path.join(base_goated_dir, predict_folder)
        new_metric = best_candidate.get("metrics", {}).get("line_of_best_fit_error")
        if new_metric is None:
            self.logger.error("New candidate lacks a valid line_of_best_fit_error metric. Not saving summary.")
            return

        existing_folders = [d for d in os.listdir(base_goated_dir) if os.path.isdir(os.path.join(base_goated_dir, d))]
        if predict_folder in existing_folders:
            summary_file = os.path.join(target_dir, "model_summary.json")
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, "r") as f:
                        existing_summary = json.load(f)
                    existing_metric = existing_summary.get("selected_best", {}).get("metrics", {}).get("line_of_best_fit_error")
                    if existing_metric is not None and new_metric < existing_metric:
                        self.logger.info(f"Updating folder '{predict_folder}': new metric {new_metric:.4f} is better than {existing_metric:.4f}")
                    else:
                        self.logger.info(f"Candidate not better than existing model in '{predict_folder}'. Skipping save.")
                        return
                except Exception as e:
                    self.logger.error(f"Error reading summary for '{predict_folder}': {e}")
        else:
            if len(existing_folders) >= 5:
                worst_folder = None
                worst_metric = None
                for folder in existing_folders:
                    folder_path = os.path.join(base_goated_dir, folder)
                    summary_file = os.path.join(folder_path, "model_summary.json")
                    if os.path.exists(summary_file):
                        try:
                            with open(summary_file, "r") as f:
                                summary_data = json.load(f)
                            metric_val = summary_data.get("selected_best", {}).get("metrics", {}).get("line_of_best_fit_error")
                            if metric_val is not None:
                                if worst_metric is None or metric_val > worst_metric:
                                    worst_metric = metric_val
                                    worst_folder = folder
                        except Exception as e:
                            self.logger.error(f"Error reading summary for folder '{folder}': {e}")
                if worst_folder is not None and new_metric < worst_metric:
                    worst_folder_path = os.path.join(base_goated_dir, worst_folder)
                    try:
                        shutil.rmtree(worst_folder_path)
                        self.logger.info(f"Removed worst predict horizon folder '{worst_folder}' with metric {worst_metric:.4f}")
                    except Exception as e:
                        self.logger.error(f"Failed to remove worst folder '{worst_folder}': {e}")
                else:
                    self.logger.info(f"New predict horizon '{predict_folder}' not better than existing worst. Skipping save.")
                    return

        os.makedirs(target_dir, exist_ok=True)
        summary = {
            "time_horizon": {"gather": gather, "predict": predict},
            "candidates": candidate_summaries,
            "selected_best": best_candidate,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
        summary_path = os.path.join(target_dir, "model_summary.json")
        try:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=4)
            self.logger.info(f"Saved model summary for predict horizon '{predict_folder}' at {summary_path}")
        except Exception as e:
            self.logger.error(f"Error saving model summary for '{predict_folder}': {e}")

    def update_goated_model_metric(self, metric_name: str, best_info: dict) -> None:
        """
        Updates global best metrics by copying best model files into a goated_models folder.
        
        :param metric_name: Name of the metric.
        :param best_info: Dictionary with best model information.
        """
        dest_relative = os.path.join("models", "goated_models", f"goated_{metric_name}")
        dest_dir = os.path.abspath(os.path.join(self.data_handler.base_data_dir, dest_relative))
        if os.path.exists(dest_dir):
            try:
                shutil.rmtree(dest_dir)
            except Exception as e:
                self.logger.error(f"Failed to clear goated {metric_name} folder: {e}")
        os.makedirs(dest_dir, exist_ok=True)
        source_dir = os.path.abspath(self._resolve_local_path(best_info["dest_stage"]))
        try:
            for file in os.listdir(source_dir):
                if file.lower().endswith(".png") or file.lower().endswith(".pt"):
                    src_file = os.path.join(source_dir, file)
                    dest_file = os.path.join(dest_dir, file)
                    shutil.copy2(src_file, dest_file)
        except Exception as e:
            self.logger.error(f"Failed to copy files for goated {metric_name} model: {e}")
        details_filename = f"goated_{metric_name}_info.json"
        self.data_handler.save_json(best_info, details_filename, stage=dest_relative)
        self.logger.info(f"Saved goated {metric_name} info to {details_filename}")

    def save_summary_table(self, results: list) -> None:
        """
        Saves the training summary table.
        
        :param results: List of training results.
        """
        analysis = ModelAnalysis(self.data_handler, model_stage="models")
        analysis.save_summary_table(results)
        self.logger.info("Training summary table saved.")

    def clear_memory(self) -> None:
        """
        Clears memory by closing figures, garbage collecting, and emptying CUDA cache if available.
        """
        plt.close('all')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Memory cleared.")

    def calculate_directional_accuracy(self, predictions, actuals):
        if len(predictions) == 0:
            return None
        correct_directions = sum(1 for pred, act in zip(predictions, actuals)
                                 if (pred >= 0 and act >= 0) or (pred < 0 and act < 0))
        return correct_directions / len(predictions)

    def calculate_percentage_over_prediction(self, predictions, actuals):
        if len(predictions) == 0:
            return None
        over_count = sum(1 for pred, act in zip(predictions, actuals) if pred > act)
        return over_count / len(predictions)

    def calculate_pearson_spearman(self, predictions, actuals):
        if len(predictions) < 2:
            return None, None
        pearson_corr, _ = pearsonr(predictions, actuals)
        spearman_corr, _ = spearmanr(predictions, actuals)
        return pearson_corr, spearman_corr

    def calculate_classification_metrics(self, predictions, actuals):
        if len(predictions) == 0:
            return None, None, None
        pred_dirs = [1 if p >= 0 else 0 for p in predictions]
        act_dirs = [1 if a >= 0 else 0 for a in actuals]
        precision = precision_score(act_dirs, pred_dirs, zero_division=0)
        recall = recall_score(act_dirs, pred_dirs, zero_division=0)
        f1 = f1_score(act_dirs, pred_dirs, zero_division=0)
        return precision, recall, f1

    def calculate_median_errors(self, predictions, actuals):
        if len(predictions) == 0:
            return None, None
        errors = np.abs(np.array(predictions) - np.array(actuals))
        median_ae = np.median(errors)
        safe_actuals = np.where(np.abs(actuals) < 1e-12, 1e-12, actuals)
        ape = np.abs(errors / safe_actuals) * 100.0
        median_ape = np.median(ape)
        return median_ae, median_ape

    def calculate_sharpe_ratio(self, predictions, actuals):
        if len(predictions) < 2:
            return None
        residuals = np.array(predictions) - np.array(actuals)
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        if std_res == 0:
            return None
        return mean_res / std_res

    def _resolve_local_path(self, stage_name: str) -> str:
        if self.data_handler.storage_mode == "s3":
            return os.path.join(self.data_handler.base_data_dir, stage_name)
        else:
            return os.path.join(self.data_handler.base_data_dir, stage_name)
