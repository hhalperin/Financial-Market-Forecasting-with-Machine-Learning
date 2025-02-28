

src/models/__init__.py:
"""
Models Module

This module exposes the following classes and functions:
- StockPredictor: A neural network for stock prediction.
- ModelManager: Handles model creation, training, and evaluation.
- ModelAnalysis: Performs analysis on trained models including metric computation and feature importance.
- ModelPipeline: Coordinates training across multiple time horizon combinations.
- TrainingConfig: Data class for training experiment configuration.
- get_experiment_configurations: Returns a list of TrainingConfig objects for experiments.
"""

from .stock_predictor import StockPredictor
from .model_manager import ModelManager
from .model_analysis import ModelAnalysis
from .model_pipeline import ModelPipeline
from .configuration import TrainingConfig, get_experiment_configurations

__all__ = [
    "StockPredictor",
    "ModelManager",
    "ModelAnalysis",
    "ModelPipeline",
    "TrainingConfig",
    "get_experiment_configurations"
]

configuration.py:
"""
Configuration Module

Defines the training configuration for model experiments using dataclasses.
This module provides a structure to define different experiment setups.
Additional fields can be added here, and defaults can be overridden via the centralized config.py.
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainingConfig:
    """
    Configuration for a single training experiment.
    """
    # Basic pipeline settings
    max_combos: int = 20000
    save_best_only: bool = True

    # Sentiment filtering settings
    filter_sentiment: bool = False
    sentiment_threshold: float = 0.2
    sentiment_cols: List[str] = field(default_factory=lambda: ["title_positive", "summary_negative"])
    sentiment_mode: str = "any"  # "any" or "all"

    # Fluctuation filtering settings
    filter_fluctuation: bool = False
    fluct_threshold: float = 1.0  # For example, remove rows with |target| < 1.0

    # Model training hyperparameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])

def get_experiment_configurations() -> List[TrainingConfig]:
    """
    Returns a list of TrainingConfig objects for different experiments.
    These configurations are optional and can be used to run multiple training experiments.
    """
    return [
        TrainingConfig(
            max_combos=50,
            filter_sentiment=True,
            sentiment_threshold=0.3,
            sentiment_cols=["title_positive", "summary_negative", "expected_sentiment"],
            sentiment_mode="any",
            filter_fluctuation=False,
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            hidden_layers=[256, 128, 64]
        ),
        TrainingConfig(
            max_combos=50,
            filter_sentiment=True,
            sentiment_threshold=0.4,
            sentiment_cols=["summary_positive", "summary_negative"],
            sentiment_mode="all",
            filter_fluctuation=True,
            fluct_threshold=2.0,
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            hidden_layers=[256, 128, 64]
        ),
        # Additional configurations can be added as needed.
    ]

cpu_optimization.py:
"""
Main Module for Model Training and Evaluation

This script orchestrates the model training pipeline:
  - Loads configuration parameters from Settings.
  - Initializes DataHandler, TimeHorizonManager, ModelManager, and ModelPipeline.
  - Saves generated time horizon combos to CSV.
  - Trains models across different time horizon combinations in parallel using CPU optimization,
    with a progress bar showing the overall horizon training progress.
  - Optionally performs hyperparameter optimization using Optuna.
"""

import os
import json
import pandas as pd
import optuna
from tqdm import tqdm
from src.utils.logger import get_logger
from src.utils.data_handler import DataHandler
from src.data_processing.time_horizon_manager import TimeHorizonManager
from .model_manager import ModelManager
from .model_analysis import ModelAnalysis
from .model_pipeline import ModelPipeline
from src.config import settings  # Global settings
from joblib import Parallel, delayed

logger = get_logger("MLTrainingMain")

def train_single_horizon(combo, X_df, df, model_params, sentiment_threshold, data_handler, index, total):
    """
    Train a model for a single horizon combination.
    Logs progress (index/total) for clarity.
    """
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize ModelManager with updated DataHandler
    model_manager = ModelManager(
        input_size=model_params["input_size"],
        hidden_layers=model_params["hidden_layers"],
        learning_rate=model_params["model_learning_rate"],
        batch_size=model_params["batch_size"],
        epochs=model_params["epochs"],
        data_handler=data_handler,
        use_time_split=True,
        model_stage="models"
    )
    model_manager.device = device

    target_col = f"{combo['predict_name']}_percentage_change"
    if target_col not in df.columns:
        logger.warning(f"[{index}/{total}] Target column {target_col} not found. Skipping combo {combo}.")
        return {"combo": combo, "mse": None}

    df_f = df.dropna(subset=[target_col])
    if df_f.empty:
        return {"combo": combo, "mse": None}

    # Apply outlier removal via IQR and z-score filtering
    q1 = df_f[target_col].quantile(0.25)
    q3 = df_f[target_col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_f = df_f[(df_f[target_col] >= lower_bound) & (df_f[target_col] <= upper_bound)]
    if df_f.empty:
        return {"combo": combo, "mse": None}
    from scipy.stats import zscore
    df_f = df_f[(abs(zscore(df_f[target_col])) < 3.0)]
    if df_f.empty:
        logger.warning(f"[{index}/{total}] All data removed by z-score filter for {target_col}. Skipping {combo}.")
        return {"combo": combo, "mse": None}

    X_f = X_df.loc[df_f.index]
    y = df_f[target_col].values
    if X_f.empty:
        return {"combo": combo, "mse": None}

    # For simplicity, no additional column filtering is applied here.
    model_name = f"horizon_{combo['gather_name']}_{combo['predict_name']}"
    mse = model_manager.train_and_evaluate(
        X_f.values,
        y,
        model_name=model_name
    )
    logger.info(f"[{index}/{total}] Finished training {model_name} with MSE = {mse:.4f}" if mse is not None else f"[{index}/{total}] {model_name} returned no MSE")
    return {"combo": combo, "mse": mse}

def main(sentiment_threshold: float = None) -> None:
    """Main function to execute the training pipeline."""
    sentiment_threshold = sentiment_threshold if sentiment_threshold is not None else settings.sentiment_threshold

    config = {
        "ticker": settings.ticker,
        "start_date": settings.start_date,
        "end_date": settings.end_date,
        "local_mode": settings.local_mode if settings.s3_bucket.strip() == "" else False,
        "storage_mode": "local" if settings.local_mode or settings.s3_bucket.strip() == "" else "s3",
        "s3_bucket": settings.s3_bucket.strip(),
        "model_learning_rate": settings.model_learning_rate,
        "model_batch_size": settings.model_batch_size,
        "model_epochs": settings.model_epochs,
        "model_hidden_layers": settings.model_hidden_layers,
    }
    logger.info(f"[MLTrainingMain] Config: {config}")

    # Initialize DataHandler
    data_handler = DataHandler(
        bucket=config["s3_bucket"],
        base_data_dir=settings.data_storage_path,
        storage_mode=config["storage_mode"]
    )
    ticker = config["ticker"]
    date_range = f"{config['start_date']}_to_{config['end_date']}"

    # Load data
    numeric_filename = data_handler.construct_filename(ticker, "numeric", date_range, "csv")
    df_numeric = data_handler.load_data(numeric_filename, data_type="csv", stage="numeric")
    preprocessed_filename = data_handler.construct_filename(ticker, "preprocessed", date_range, "csv")
    df = data_handler.load_data(preprocessed_filename, data_type="csv", stage="preprocessed")

    if df_numeric is None or df is None or df_numeric.empty or df.empty:
        logger.error("Missing or empty data. Check previous stages.")
        return

    logger.info(f"Loaded numeric data shape: {df_numeric.shape}, processed data shape: {df.shape}")
    X_df = df_numeric.select_dtypes(include=["number"])
    logger.info(f"Training features shape: {X_df.shape}")

    horizon_manager = TimeHorizonManager(
        max_gather_minutes=settings.max_gather_minutes,
        max_predict_minutes=settings.max_predict_minutes,
        step=settings.time_horizon_step
    )
    combos = horizon_manager.generate_horizon_combos()
    combos = horizon_manager.filter_combos(combos, max_combos=settings.num_combos)
    total_combos = len(combos)
    logger.info(f"Total horizon combos count: {total_combos}")

    horizons_dir = os.path.join(settings.data_storage_path, "models", "horizons")
    os.makedirs(horizons_dir, exist_ok=True)
    pd.DataFrame(combos).to_csv(os.path.join(horizons_dir, "horizons.csv"), index=False)

    model_params = {
        "input_size": X_df.shape[1],
        "model_learning_rate": config["model_learning_rate"],
        "batch_size": config["model_batch_size"],
        "epochs": config["model_epochs"],
        "hidden_layers": config["model_hidden_layers"],
    }

    # Use CPU optimization with joblib Parallel and a progress bar
    logger.info("Starting parallel training across horizons...")
    results = Parallel(n_jobs=mp.cpu_count()-2)(
        delayed(train_single_horizon)(combo, X_df, df, model_params, sentiment_threshold, data_handler, i+1, total_combos)
        for i, combo in enumerate(tqdm(combos, desc="Horizon Progress", unit="horizon"))
    )
    logger.info("Finished parallel training across horizons.")
    for res in results:
        mse_str = f"{res['mse']:.4f}" if res['mse'] is not None else "None"
        logger.info(f"Combo {res['combo']['gather_name']} to {res['combo']['predict_name']}: MSE = {mse_str}")

    # Hyperparameter optimization with Optuna (optional)
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_layers = [trial.suggest_int(f"n_units_l{i}", 32, 512, step=32) for i in range(n_layers)]
        batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

        opt_model_manager = ModelManager(
            input_size=X_df.shape[1],
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=settings.model_epochs,
            data_handler=data_handler,
            use_time_split=True,
            model_stage="models"
        )
        target_column = f"{combos[0]['predict_name']}_percentage_change" if combos else f"{settings.ticker}_percentage_change"
        mse = opt_model_manager.train_and_evaluate(
            X_df.values,
            df[target_column].values,
            model_name="optuna_trial",
            trial=trial
        )
        return mse

    logger.info("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction="minimize", storage="sqlite:///optuna.db", study_name="my_study", load_if_exists=True)
    study.optimize(objective, n_trials=settings.hyperparameter_trials)
    logger.info(f"Best trial: {study.best_trial.number} with MSE: {study.best_trial.value:.4f}")
    logger.info(f"Best parameters: {study.best_trial.params}")

if __name__ == "__main__":
    main()

main.py:
"""
Main Module for Model Training and Evaluation

This script orchestrates the model training pipeline:
  - Loads configuration parameters from Settings.
  - Initializes a DataHandler to load preprocessed and numeric data.
  - Initializes the TimeHorizonManager, ModelManager, and ModelPipeline.
  - Saves generated time horizon combos to CSV.
  - Trains models across different time horizon combinations sequentially.
  - Optionally performs hyperparameter optimization using Optuna.
"""

import os
import json
import pandas as pd
import optuna
from tqdm import tqdm
from src.utils.logger import get_logger
from src.utils.data_handler import DataHandler
from src.data_processing.time_horizon_manager import TimeHorizonManager
from .model_manager import ModelManager
from .model_analysis import ModelAnalysis
from .model_pipeline import ModelPipeline
from src.config import settings  # Global settings

logger = get_logger("MLTrainingMain")

def main(sentiment_threshold: float = None) -> None:
    """Main function to execute the sequential model training pipeline."""
    sentiment_threshold = sentiment_threshold if sentiment_threshold is not None else settings.sentiment_threshold

    # Configuration dictionary
    config = {
        "ticker": settings.ticker,
        "start_date": settings.start_date,
        "end_date": settings.end_date,
        "local_mode": settings.local_mode if settings.s3_bucket.strip() == "" else False,
        "storage_mode": "local" if settings.local_mode or settings.s3_bucket.strip() == "" else "s3",
        "s3_bucket": settings.s3_bucket.strip(),
        "model_learning_rate": settings.model_learning_rate,
        "model_batch_size": settings.model_batch_size,
        "model_epochs": settings.model_epochs,
        "model_hidden_layers": settings.model_hidden_layers,
    }
    logger.info(f"[MLTrainingMain] Config: {config}")

    # Initialize DataHandler (ensuring updated version with save_figure_bytes is used)
    data_handler = DataHandler(
        bucket=config["s3_bucket"],
        base_data_dir=settings.data_storage_path,
        storage_mode=config["storage_mode"]
    )
    ticker = config["ticker"]
    date_range = f"{config['start_date']}_to_{config['end_date']}"

    # Load numeric and preprocessed data.
    numeric_filename = data_handler.construct_filename(ticker, "numeric", date_range, "csv")
    df_numeric = data_handler.load_data(numeric_filename, data_type="csv", stage="numeric")
    preprocessed_filename = data_handler.construct_filename(ticker, "preprocessed", date_range, "csv")
    df = data_handler.load_data(preprocessed_filename, data_type="csv", stage="preprocessed")

    # Validate data loading.
    if df_numeric is None or df is None or df_numeric.empty or df.empty:
        logger.error("Missing or empty numeric or preprocessed data. Check previous stages.")
        return

    logger.info(f"Loaded numeric data shape: {df_numeric.shape}, raw processed data shape: {df.shape}")

    # Select numeric columns for training.
    X_df = df_numeric.select_dtypes(include=["number"])
    logger.info(f"Numeric DataFrame for training: shape={X_df.shape}")

    # Initialize TimeHorizonManager.
    horizon_manager = TimeHorizonManager(
        max_gather_minutes=settings.max_gather_minutes,
        max_predict_minutes=settings.max_predict_minutes,
        step=settings.time_horizon_step
    )
    # Generate and filter time horizon combinations.
    combos = horizon_manager.generate_horizon_combos()
    combos = horizon_manager.filter_combos(combos, max_combos=settings.num_combos)
    logger.info(f"Time horizon combos count: {len(combos)}")

    # Save all time horizon combos to CSV for verification.
    horizons_dir = os.path.join(settings.data_storage_path, "models", "horizons")
    os.makedirs(horizons_dir, exist_ok=True)
    combos_df = pd.DataFrame(combos)
    combos_csv_path = os.path.join(horizons_dir, "horizons.csv")
    combos_df.to_csv(combos_csv_path, index=False)

    # Prepare model parameters.
    model_params = {
        "input_size": X_df.shape[1],
        "hidden_layers": config["model_hidden_layers"],
        "learning_rate": config["model_learning_rate"],
        "batch_size": config["model_batch_size"],
        "epochs": config["model_epochs"],
    }

    # Initialize ModelManager and ModelPipeline.
    model_manager = ModelManager(
        input_size=X_df.shape[1],
        hidden_layers=config["model_hidden_layers"],
        learning_rate=config["model_learning_rate"],
        batch_size=config["model_batch_size"],
        epochs=config["model_epochs"],
        data_handler=data_handler,
        use_time_split=True
    )
    pipeline = ModelPipeline(model_manager, data_handler, horizon_manager)

    # Define sentiment columns.
    sentiment_cols = ["title_positive", "summary_positive", "expected_sentiment", "summary_negative"]

    # Sequentially train models across horizon combinations.
    logger.info("Starting sequential training across horizon combinations.")
    results = []
    for i, combo in enumerate(tqdm(combos, desc="Training Horizons", unit="horizon")):
        # Process each combo one after the other.
        # We call the pipeline's train_on_horizons method on a single combo list.
        res = pipeline.train_on_horizons(
            X_df,
            df,  # Raw DataFrame for target columns.
            max_combos=1,
            save_best_only=True,
            filter_sentiment=settings.filter_sentiment,
            sentiment_threshold=sentiment_threshold,
            sentiment_cols=sentiment_cols,
            sentiment_mode="any",
            combos=[combo]
        )
        if res:
            results.extend(res)
        logger.info(f"[{i+1}/{len(combos)}] Processed combo: {combo['gather_name']} to {combo['predict_name']}.")

    logger.info("Finished sequential horizon training pipeline.")
    for result in results:
        mse_str = f"{result['mse']:.4f}" if result['mse'] is not None else "None"
        logger.info(f"Combo {result['gather_horizon']} to {result['predict_horizon']}: MSE = {mse_str}")

    # Optional hyperparameter optimization with Optuna.
    def objective(trial):
        """Objective function for Optuna to optimize hyperparameters."""
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_layers = [trial.suggest_int(f"n_units_l{i}", 32, 512, step=32) for i in range(n_layers)]
        batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

        opt_model_manager = ModelManager(
            input_size=X_df.shape[1],
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=settings.model_epochs,
            data_handler=data_handler,
            use_time_split=True,
            seq_length=None
        )
        target_column = f"{combos[0]['predict_name']}_percentage_change" if combos else f"{settings.ticker}_percentage_change"
        mse = opt_model_manager.train_and_evaluate(
            X_df.values,
            df[target_column].values,
            model_name="optuna_trial",
            trial=trial
        )
        return mse

    logger.info("Starting hyperparameter optimization with Optuna for architecture search...")
    study = optuna.create_study(direction="minimize", storage="sqlite:///optuna.db", study_name="my_study", load_if_exists=True)
    study.optimize(objective, n_trials=settings.hyperparameter_trials)
    logger.info(f"Best trial: {study.best_trial.number} with MSE: {study.best_trial.value:.4f}")
    logger.info(f"Best parameters: {study.best_trial.params}")

if __name__ == "__main__":
    main()

model_analysis.py:
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
from src.utils.data_handler import DataHandler
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

    def save_figure_bytes(self, bytes_data: bytes, filename: str, stage: str = None):
        # Determine the directory to save the figure
        if stage:
            stage_path = os.path.join(self.data_handler.base_data_dir, stage)
        else:
            stage_path = self.data_handler.base_data_dir
        os.makedirs(stage_path, exist_ok=True)
        
        file_path = os.path.join(stage_path, filename)
        try:
            with open(file_path, "wb") as f:
                f.write(bytes_data)
            #print(f"Saved figure to {file_path}")
        except Exception as e:
            print(f"Error saving figure to {file_path}: {e}")

    def _save_figure(self, figure_filename: str) -> None:
        try:
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=100)
            buffer.seek(0)
            # Check if save_figure_bytes exists and is callable
            if hasattr(self.data_handler, "save_figure_bytes") and callable(self.data_handler.save_figure_bytes):
                self.data_handler.save_figure_bytes(buffer.read(), figure_filename, stage=self.model_stage)
            else:
                self.logger.error(f"DataHandler does not implement 'save_figure_bytes'. Cannot save {figure_filename}.")
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

    def compute_feature_correlations(self, X_df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        df = X_df.copy()
        df['target'] = y
        return df.corr()

model_manager.py:
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import numpy as np
import pandas as pd
from .stock_predictor import StockPredictor  # Assuming using the feedforward predictor
from .model_analysis import ModelAnalysis
from src.utils.logger import get_logger
from typing import Tuple, Any, List, Optional
from src.config import settings

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_loss = np.inf
        self.should_stop = False

    def __call__(self, val_loss: float) -> None:
        if val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

class ModelManager:
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = None,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        data_handler: Any = None,
        model_stage: str = "models",
        use_time_split: bool = True
    ) -> None:
        self.input_size = input_size
        self.hidden_layers = hidden_layers or [256, 128, 64]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_handler = data_handler
        self.model_stage = model_stage
        self.use_time_split = use_time_split
        self.logger = get_logger(self.__class__.__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.logger.info(f"Using device: {self.device}")

    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, model_name: str, trial=None) -> Tuple[Any, dict]:
        # Create train/validation/test splits
        if self.use_time_split:
            if hasattr(X, "index"):
                X = X.sort_index()
                y = y[X.index]
            train_size = int(len(X) * 0.7)
            val_size = int(len(X) * 0.85)
            X_train, X_val, X_test = X[:train_size], X[train_size:val_size], X[val_size:]
            y_train, y_val, y_test = y[:train_size], y[train_size:val_size], y[val_size:]
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        if trial:
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
            n_layers = trial.suggest_int("n_layers", 1, 3)
            hidden_layers = [trial.suggest_int(f"n_units_l{i}", 32, 512, step=32) for i in range(n_layers)]
            batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
            dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        else:
            learning_rate = self.learning_rate
            hidden_layers = self.hidden_layers
            batch_size = self.batch_size
            dropout_rate = settings.model_dropout_rate

        train_loader = self._get_dataloader(X_train, y_train, batch_size)
        val_loader = self._get_dataloader(X_val, y_val, batch_size)

        model = StockPredictor(self.input_size, hidden_layers, dropout_rate=dropout_rate).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=settings.model_weight_decay)
        loss_fn = nn.SmoothL1Loss() if settings.model_loss_function.lower() == "smoothl1" else nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=settings.lr_scheduler_factor, patience=settings.lr_scheduler_patience
        )

        # Train model (only final epoch info will be logged)
        training_history, best_model_state, best_val_mse = self._train_model(
            model, train_loader, val_loader, optimizer, loss_fn, scheduler, model_name
        )

        if trial:
            return best_val_mse

        model.load_state_dict(best_model_state)
        y_test_pred = self._predict(model, X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        evs = explained_variance_score(y_test, y_test_pred)
        eps = 1e-8
        denom = np.where(np.abs(y_test) < eps, eps, y_test)
        mape = np.mean(np.abs((y_test - y_test_pred) / denom)) * 100
        tol = getattr(settings, "regression_accuracy_tolerance", 0.1)
        relative_errors = np.abs((y_test - y_test_pred) / denom)
        regression_accuracy = np.mean(relative_errors < tol) * 100
        slope, intercept = np.polyfit(y_test, y_test_pred, 1)
        line_of_best_fit_error = abs(slope - 1) + abs(intercept)

        # Additional metrics using ModelAnalysis helpers
        from .model_summary import ModelSummary
        temp_summary = ModelSummary(self.data_handler)
        directional_accuracy = temp_summary.calculate_directional_accuracy(y_test_pred, y_test)
        percentage_over_prediction = temp_summary.calculate_percentage_over_prediction(y_test_pred, y_test)
        pearson_corr, spearman_corr = temp_summary.calculate_pearson_spearman(y_test_pred, y_test)

        analysis = ModelAnalysis(self.data_handler, model_stage=self.model_stage)
        analysis.generate_all_plots(
            training_history=training_history,
            y_true=y_test,
            y_pred=y_test_pred,
            model=model,
            X_np=X_test,
            model_name=model_name,
            value=test_mse
        )

        self.logger.info(f"Final metrics for '{model_name}': MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, "
                         f"R²: {test_r2:.4f}, Explained Variance: {evs:.4f}, MAPE: {mape:.2f}%, "
                         f"Regression Accuracy: {regression_accuracy:.2f}%, Line Fit Error: {line_of_best_fit_error:.4f}")

        metrics = {
            "mse": test_mse,
            "mae": test_mae,
            "r2": test_r2,
            "explained_variance": evs,
            "mape": mape,
            "regression_accuracy": regression_accuracy,
            "line_of_best_fit_error": line_of_best_fit_error,
            "directional_accuracy": directional_accuracy,
            "percentage_over_prediction": percentage_over_prediction,
            "pearson_correlation": pearson_corr,
            "spearman_correlation": spearman_corr,
            "training_history": training_history,
        }
        return model, metrics

    def _train_model(self, model, train_loader, val_loader, optimizer, loss_fn, scheduler, model_name: str) -> Tuple[dict, dict, float]:
        """
        Train the model over multiple epochs and log only the final epoch's performance.
        Early stopping is applied and the final (or stopped) epoch's summary is logged.
        """
        training_history = {"train_loss": [], "val_loss": []}
        early_stopper = EarlyStopping(patience=5, min_delta=0.0)
        best_model_state = None
        best_val_mse = float("inf")
        final_epoch = None

        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = loss_fn(y_pred.view(-1), y_batch.view(-1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_train_loss = running_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            y_val_true = []
            y_val_pred = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    preds = model(X_batch)
                    loss = loss_fn(preds.view(-1), y_batch.view(-1))
                    val_loss += loss.item()
                    y_val_true.append(y_batch.cpu().numpy().reshape(-1))
                    y_val_pred.append(preds.cpu().numpy().reshape(-1))
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            y_val_true = np.concatenate(y_val_true)
            y_val_pred = np.concatenate(y_val_pred)
            val_mse = np.mean((y_val_true - y_val_pred) ** 2)

            training_history["train_loss"].append(avg_train_loss)
            training_history["val_loss"].append(avg_val_loss)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_model_state = model.state_dict()
            early_stopper(avg_val_loss)
            final_epoch = epoch + 1
            if early_stopper.should_stop:
                break

        # Log only final epoch summary
        self.logger.debug(f"Final Epoch {final_epoch}/{self.epochs} for '{model_name}' -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MSE: {val_mse:.4f}")
        return training_history, best_model_state, best_val_mse

    def _predict(self, model, X: np.ndarray) -> np.ndarray:
        model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = model(X_t)
        return preds.view(-1).cpu().numpy()

    def _get_dataloader(self, X: np.ndarray, y: np.ndarray, batch_size: int = None) -> DataLoader:
        batch_size = batch_size or self.batch_size
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=True
        )

    def save_model(self, model: nn.Module, filepath: str) -> None:
        torch.save(model.state_dict(), filepath)

model_pipeline.py:
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
        Initialize with model manager, data handler, and horizon manager.
        """
        self.model_manager = model_manager
        self.data_handler = data_handler
        self.horizon_manager = horizon_manager
        self.logger = get_logger(self.__class__.__name__)
        self.summary_manager = ModelSummary(data_handler=self.data_handler)

    def advanced_filter_sentiment(self, df: pd.DataFrame, sentiment_columns: list = None,
                                  threshold: float = 0.2, mode: str = "any") -> pd.DataFrame:
        """
        Filter dataframe rows based on sentiment column thresholds.
        """
        if not sentiment_columns:
            return df
        valid_cols = [col for col in sentiment_columns if col in df.columns]
        if not valid_cols:
            self.logger.warning("No valid sentiment columns found. Returning df unchanged.")
            return df
        orig_len = len(df)
        bool_masks = [df[col] >= threshold for col in valid_cols]
        combined_mask = bool_masks[0]
        for mask in bool_masks[1:]:
            combined_mask = combined_mask | mask if mode == "any" else combined_mask & mask
        filtered_df = df[combined_mask].copy()
        self.logger.info(f"Sentiment filter: reduced rows from {orig_len} to {len(filtered_df)} "
                         f"(mode='{mode}', threshold={threshold}, columns={valid_cols}).")
        return filtered_df

    def _drop_future_horizon_columns(self, X: pd.DataFrame, gather_horizon_str: str) -> pd.DataFrame:
        """
        Keep only columns with minute values less than or equal to the gather horizon.
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
        Return True if the average absolute change in the target column is below threshold.
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
        Return the full path for a given stage.
        """
        return os.path.join(self.data_handler.base_data_dir, stage_name)

    def _recursive_update_goated(self, metric_updates: List[tuple]) -> None:
        """
        Recursively update global best metrics.
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
        Update global best metrics if current candidate is better.
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

            #self.logger.info(f"Training with {X_f_numeric.shape[1]} numeric features: {list(X_f_numeric.columns)}. Target Column: {target_col}")
            candidate_model_name = f"model_{gather}_to_{predict}"
            self.logger.info(f"Training model: {candidate_model_name}")

            # Determine destination folder based on config flag.
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

            # Directly set model stage to destination; no temp directories are used.
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
                # Save training data CSV directly in goated_models folder.
                self.data_handler.save_dataframe(train_df, training_data_filename, stage=dest_folder_relative)
                self.logger.info(f"Saved training data to {training_data_filename} in stage {dest_folder_relative}.")

                # Save model summary JSON using ModelSummary (destination already in goated_models)
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

model_summary.py:
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
from .model_analysis import ModelAnalysis  # Assuming this is where your analysis code lives
from src.config import settings

class ModelSummary:
    def __init__(self, data_handler):
        """
        Manage model summary logic, track global 'goated' (best) models,
        and compute additional metrics.
        """
        self.data_handler = data_handler
        self.logger = get_logger(self.__class__.__name__)

        # Global best for standard metrics.
        self.global_best_mse = float("inf")
        self.global_best_info_mse = None

        self.global_best_mae = float("inf")
        self.global_best_info_mae = None

        self.global_best_r2 = -float("inf")
        self.global_best_info_r2 = None

        self.global_best_lobf = float("inf")
        self.global_best_info_lobf = None

        # Additional suggested metrics:
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

        # New global best fields for additional metrics:
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
        Save the JSON summary for a given predict horizon.
        Only one folder per predict horizon is maintained in goated_models.
        If the folder already exists, update it only if the new candidate (using line_of_best_fit_error)
        is better. Also, limit the total number of predict horizon folders to 5.
        """
        # Define the base goated folder.
        base_goated_dir = os.path.join(self.data_handler.base_data_dir, "models", "goated_models")
        os.makedirs(base_goated_dir, exist_ok=True)

        # Use the predict string as the folder name, e.g. "22_minutes"
        predict_folder = predict
        target_dir = os.path.join(base_goated_dir, predict_folder)

        new_metric = best_candidate.get("metrics", {}).get("line_of_best_fit_error")
        if new_metric is None:
            self.logger.error("New candidate lacks a valid line_of_best_fit_error metric. Not saving summary.")
            return

        # Get list of existing predict horizon folders.
        existing_folders = [d for d in os.listdir(base_goated_dir) if os.path.isdir(os.path.join(base_goated_dir, d))]

        if predict_folder in existing_folders:
            # Folder exists: load its summary and compare.
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
            # Folder does not exist.
            if len(existing_folders) >= 5:
                # Determine the worst folder among the existing ones.
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

        # Create or update the target folder.
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
        Update global best metrics by copying best model files into a goated_models folder.
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

    def save_summary(self, results: list) -> None:
        """Save the training summary table using ModelAnalysis."""
        analysis = ModelAnalysis(self.data_handler, model_stage="models")
        analysis.save_summary_table(results)
        self.logger.info("Training summary table saved.")

    def clear_memory(self) -> None:
        """Clear memory by closing figures, collecting garbage, and emptying CUDA cache if available."""
        plt.close('all')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Memory cleared.")

    def calculate_directional_accuracy(self, predictions, actuals):
        if len(predictions) == 0:
            return None
        correct_directions = 0
        for pred, act in zip(predictions, actuals):
            if (pred >= 0 and act >= 0) or (pred < 0 and act < 0):
                correct_directions += 1
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
        """
        Resolve the local path for saving files based on the data_handler's storage mode.
        """
        if self.data_handler.storage_mode == "s3":
            return os.path.join(self.data_handler.base_data_dir, stage_name)
        else:
            return os.path.join(self.data_handler.base_data_dir, stage_name)

stock_predictor.py:
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class StockPredictor(nn.Module):
    """Feedforward neural network for stock price prediction."""
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int = 1, dropout_rate: float = 0.2) -> None:
        super(StockPredictor, self).__init__()
        layers = []
        in_features = input_size
        for units in hidden_layers:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = units
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class StockCNNPredictor(nn.Module):
    """1D CNN model for stock price prediction with sequence input."""
    def __init__(self, input_size: int, seq_length: int, dropout_rate: float = 0.2):
        super(StockCNNPredictor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        # Calculate the size after convolutions and pooling: seq_length // 2 due to one pooling layer
        self.fc1 = nn.Linear(128 * (seq_length // 2), 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, seq_length, features]
        x = x.transpose(1, 2)  # [batch, features, seq_length] for Conv1d
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x    