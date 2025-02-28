"""
CPU Optimization Module for Model Training and Evaluation

This script orchestrates the model training pipeline:
  - Loads configuration parameters from Settings.
  - Initializes DataHandler, TimeHorizonManager, ModelManager, and ModelPipeline.
  - Saves generated time horizon combos to CSV.
  - Trains models across different time horizon combinations in parallel using CPU optimization,
    with a progress bar showing overall horizon training progress.
  - Optionally performs hyperparameter optimization using Optuna.
"""

import os
import json
import pandas as pd
import optuna
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp

from src.utils.logger import get_logger
from src.utils.data_handler import DataHandler
from src.data_processing.time_horizon_manager import TimeHorizonManager
from .model_manager import ModelManager
from .model_analysis import ModelAnalysis
from .model_pipeline import ModelPipeline
from src.config import settings  # Global settings

logger = get_logger("MLTrainingMain")

def train_single_horizon(combo, X_df, df, model_params, sentiment_threshold, data_handler, index, total):
    """
    Trains a model for a single horizon combination and returns its MSE.
    
    :param combo: Dictionary with gather and predict horizon information.
    :param X_df: DataFrame of features.
    :param df: DataFrame containing target columns.
    :param model_params: Dictionary of model hyperparameters.
    :param sentiment_threshold: Threshold for sentiment filtering.
    :param data_handler: DataHandler instance.
    :param index: Current index in training loop.
    :param total: Total number of horizon combinations.
    :return: Dictionary with the combo and its MSE.
    """
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    model_name = f"horizon_{combo['gather_name']}_{combo['predict_name']}"
    mse = model_manager.train_and_evaluate(
        X_f.values,
        y,
        model_name=model_name
    )
    if mse is not None:
        logger.info(f"[{index}/{total}] Finished training {model_name} with MSE = {mse:.4f}")
    else:
        logger.info(f"[{index}/{total}] {model_name} returned no MSE")
    return {"combo": combo, "mse": mse}

def main(sentiment_threshold: float = None) -> None:
    """
    Executes the parallel training pipeline.
    
    :param sentiment_threshold: Optional sentiment threshold override.
    """
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

    data_handler = DataHandler(
        bucket=config["s3_bucket"],
        base_data_dir=settings.data_storage_path,
        storage_mode=config["storage_mode"]
    )
    ticker = config["ticker"]
    date_range = f"{config['start_date']}_to_{config['end_date']}"

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

    logger.info("Starting parallel training across horizons...")
    results = Parallel(n_jobs=mp.cpu_count()-2)(
        delayed(train_single_horizon)(combo, X_df, df, model_params, sentiment_threshold, data_handler, i+1, total_combos)
        for i, combo in enumerate(tqdm(combos, desc="Horizon Progress", unit="horizon"))
    )
    logger.info("Finished parallel training across horizons.")
    for res in results:
        mse_str = f"{res['mse']:.4f}" if res['mse'] is not None else "None"
        logger.info(f"Combo {res['combo']['gather_name']} to {res['combo']['predict_name']}: MSE = {mse_str}")

    def objective(trial):
        """Objective function for Optuna hyperparameter tuning."""
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

    logger.info("Starting hyperparameter optimization with Optuna for architecture search...")
    study = optuna.create_study(direction="minimize", storage="sqlite:///optuna.db", study_name="my_study", load_if_exists=True)
    study.optimize(objective, n_trials=settings.hyperparameter_trials)
    logger.info(f"Best trial: {study.best_trial.number} with MSE: {study.best_trial.value:.4f}")
    logger.info(f"Best parameters: {study.best_trial.params}")

if __name__ == "__main__":
    main()
