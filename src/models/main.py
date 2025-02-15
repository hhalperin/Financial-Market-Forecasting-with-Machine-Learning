"""
Main Module for Model Training and Evaluation

This script orchestrates the model training pipeline:
  - Loads configuration parameters from the centralized Settings.
  - Builds a DataHandler to load preprocessed and numeric data.
  - Initializes the ModelManager and ModelPipeline.
  - Saves all generated time horizon combos to a CSV for verification.
  - Trains models across different time horizon combinations.
  - Optionally performs hyperparameter optimization (including architecture search) using Optuna.
  - Launches a live dashboard (via Streamlit) in a separate process so that training progress is updated live.
"""

import os
import json
import time
import pandas as pd
import optuna
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from src.utils.logger import get_logger
from src.utils.data_handler import DataHandler
from src.data_processing.time_horizon_manager import TimeHorizonManager
from .model_manager import ModelManager
from .model_analysis import ModelAnalysis
from .model_pipeline import ModelPipeline
from src.config import settings  # Global settings

logger = get_logger("MLTrainingMain")


def main(sentiment_threshold: float = None) -> None:
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

    # Initialize DataHandler.
    data_handler = DataHandler(
        bucket=config["s3_bucket"],
        base_data_dir=settings.data_storage_path,
        storage_mode=config["storage_mode"]
    )
    ticker = config["ticker"]
    date_range = f"{config['start_date']}_to_{config['end_date']}"

    # Load data.
    numeric_filename = data_handler.construct_filename(ticker, "numeric", date_range, "csv")
    df_numeric = data_handler.load_data(numeric_filename, data_type="csv", stage="numeric")
    preprocessed_filename = data_handler.construct_filename(ticker, "preprocessed", date_range, "csv")
    df = data_handler.load_data(preprocessed_filename, data_type="csv", stage="preprocessed")

    if df_numeric is None or df is None or df_numeric.empty or df.empty:
        logger.error("Missing or empty numeric or preprocessed data. Check previous stages.")
        return

    logger.info(f"Loaded numeric data shape: {df_numeric.shape}, raw processed data shape: {df.shape}")

    # Select numeric columns.
    X_df = df_numeric.select_dtypes(include=["number"])
    logger.info(f"Numeric DataFrame for training: shape={X_df.shape}")

    # Initialize TimeHorizonManager.
    horizon_manager = TimeHorizonManager(
        max_gather_minutes=settings.max_gather_minutes,
        max_predict_minutes=settings.max_predict_minutes,
        step=settings.time_horizon_step
    )
    # Generate and filter combos.
    combos = horizon_manager.generate_horizon_combos()
    combos = horizon_manager.filter_combos(combos, max_combos=settings.num_combos)
    logger.info(f"Time horizon combos count: {len(combos)}")

    # Save all time horizon combos to CSV for verification.
    horizons_dir = os.path.join(settings.data_storage_path, "models", "horizons")
    os.makedirs(horizons_dir, exist_ok=True)
    combos_df = pd.DataFrame(combos)
    combos_csv_path = os.path.join(horizons_dir, "horizons.csv")
    combos_df.to_csv(combos_csv_path, index=False)

    # Initialize ModelManager.
    model_manager = ModelManager(
        input_size=X_df.shape[1],
        hidden_layers=config["model_hidden_layers"],
        learning_rate=config["model_learning_rate"],
        batch_size=config["model_batch_size"],
        epochs=config["model_epochs"],
        data_handler=data_handler,
        use_time_split=True
    )
    # Initialize ModelPipeline.
    pipeline = ModelPipeline(model_manager, data_handler, horizon_manager)

    # Define sentiment columns.
    sentiment_cols = ["title_positive", "summary_positive", "expected_sentiment", "summary_negative"]
    pipeline.train_on_horizons(
        X_df,
        df,  # Raw DataFrame for target columns.
        max_combos=settings.num_combos,
        save_best_only=True,
        filter_sentiment=settings.filter_sentiment,
        sentiment_threshold=sentiment_threshold,
        sentiment_cols=sentiment_cols,
        sentiment_mode="any"
    )
    logger.info("Finished direct horizon training pipeline.")

    # Optional hyperparameter optimization using Optuna.
    def objective(trial):
        """
        Objective function for Optuna to optimize model architecture and other hyperparameters.
        """
        # The train_and_evaluate method in ModelManager uses the trial to set hyperparameters.
        mse = model_manager.train_and_evaluate(
            X_df.values,
            df[f"{settings.ticker}_percentage_change"].values,
            model_name="optuna_trial",
            trial=trial
        )
        return mse

    logger.info("Starting hyperparameter optimization with Optuna for architecture search...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=settings.hyperparameter_trials)
    logger.info(f"Best trial: {study.best_trial.number} with MSE: {study.best_trial.value:.4f}")
    logger.info(f"Best parameters: {study.best_trial.params}")


if __name__ == "__main__":
    main()