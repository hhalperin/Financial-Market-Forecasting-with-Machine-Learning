"""
Main Module for Model Training and Evaluation

This script orchestrates the model training pipeline:
  - Loads configuration parameters from the centralized Settings.
  - Builds a DataHandler to load preprocessed and numeric data from the organized data directory.
  - Initializes the ModelManager and ModelPipeline.
  - Trains models across different time horizon combinations.
  - Optionally performs hyperparameter optimization using Optuna.
"""

import os
import json
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import optuna
from sklearn.metrics import mean_squared_error
from src.utils.logger import get_logger
from src.utils.data_handler import DataHandler
from src.data_processing.time_horizon_manager import TimeHorizonManager
from .model_manager import ModelManager
from .model_analysis import ModelAnalysis
from .model_pipeline import ModelPipeline
from src.config import settings  # Import the global settings instance

logger = get_logger("MLTrainingMain")

def main(sentiment_threshold: float = None) -> None:
    # Use the global settings for configuration.
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

    # Use DataHandler with base_data_dir set to settings.data_storage_path (the organized "data" directory)
    data_handler = DataHandler(
        bucket=config["s3_bucket"],
        base_data_dir=settings.data_storage_path,
        storage_mode=config["storage_mode"]
    )
    ticker = config["ticker"]
    date_range = f"{config['start_date']}_to_{config['end_date']}"

    # Load the numeric DataFrame saved from the data processing stage.
    numeric_filename = data_handler.construct_filename(ticker, "numeric", date_range, "csv")
    df_numeric = data_handler.load_data(numeric_filename, data_type="csv", stage="numeric")

    # Load the preprocessed raw DataFrame for target columns.
    preprocessed_filename = data_handler.construct_filename(ticker, "preprocessed", date_range, "csv")
    df = data_handler.load_data(preprocessed_filename, data_type="csv", stage="preprocessed")

    if df_numeric is None or df is None or df_numeric.empty or df.empty:
        logger.error("Missing or empty numeric or preprocessed data. Check previous stages.")
        return

    logger.info(f"Loaded numeric data shape: {df_numeric.shape}, raw processed data shape: {df.shape}")

    # Select only numeric columns for training.
    X_df = df_numeric.select_dtypes(include=["number"])
    logger.info(f"Numeric DataFrame for training (after selecting numeric columns): shape={X_df.shape}")

    horizon_manager = TimeHorizonManager(
        max_gather_minutes=settings.max_gather_minutes,
        max_predict_minutes=settings.max_predict_minutes,
        step=settings.time_horizon_step
    )
    combos = horizon_manager.generate_horizon_combos()
    combos = horizon_manager.filter_combos(combos, max_combos=2000)
    logger.info(f"Time horizon combos count: {len(combos)}")

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

    # Define sentiment columns for filtering.
    sentiment_cols = ["title_positive", "summary_positive", "expected_sentiment", "summary_negative"]
    pipeline.train_on_horizons(
        X_df,
        df,  # Raw DataFrame used for target columns.
        max_combos=10,
        save_best_only=True,
        filter_sentiment=True,
        sentiment_threshold=sentiment_threshold,
        sentiment_cols=sentiment_cols,
        sentiment_mode="any"
    )
    logger.info("Finished direct horizon training pipeline.")

    # Optional hyperparameter optimization with Optuna.
    def train_single_horizon(combo_dict, X_df, df):
        single_combo = [combo_dict]
        results = pipeline.train_on_horizons(
            X_df,
            df,
            max_combos=1,
            save_best_only=True,
            filter_sentiment=True,
            sentiment_threshold=sentiment_threshold,
            sentiment_cols=sentiment_cols,
            sentiment_mode="any"
        )
        if not results:
            return float("inf")
        return results[0]["mse"]

    def optimize(trial):
        import json
        selected_combo_str = trial.suggest_categorical(
            "horizon_combo",
            [
                json.dumps({
                    "gather_name": c["gather_name"],
                    "gather_td": c["gather_td"].total_seconds(),
                    "predict_name": c["predict_name"],
                    "predict_td": c["predict_td"].total_seconds()
                }) for c in combos
            ]
        )
        combo_dict = json.loads(selected_combo_str)
        target_col = f"{combo_dict['predict_name']}_change"
        if target_col not in df.columns:
            logger.warning(f"Missing target {target_col}, skipping combo.")
            return float("inf")
        mse_val = train_single_horizon(combo_dict, X_df, df)
        return mse_val

    logger.info("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction="minimize")
    def run_study_with_progress(study, n_trials):
        with tqdm(total=n_trials, desc="Optuna Trials") as pbar:
            def callback(study, trial):
                pbar.update(1)
            study.optimize(optimize, n_trials=n_trials, callbacks=[callback])
    run_study_with_progress(study, n_trials=5)
    logger.info(f"Best trial: {study.best_trial.number} with MSE: {study.best_trial.value:.4f}")
    logger.info(f"Best parameters: {study.best_trial.params}")

if __name__ == "__main__":
    main()
