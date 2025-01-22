# src/models/main.py

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

logger = get_logger("MLTrainingMain")

def load_config():
    """
    Load environment variables for ticker/date range, local vs. S3, etc.
    """
    load_dotenv()
    local_mode = os.getenv("LOCAL_MODE", "false").lower() == "true"
    storage_mode = "local" if local_mode else "s3"
    s3_bucket = os.getenv("S3_BUCKET") if not local_mode else None

    return {
        "ticker": os.getenv("TICKER", "NVDA"),
        "start_date": os.getenv("START_DATE", "2022-01-01"),
        "end_date": os.getenv("END_DATE", "2024-01-31"),
        "local_mode": local_mode,
        "storage_mode": storage_mode,
        "s3_bucket": s3_bucket,
    }

def build_data_handler(config):
    if config["storage_mode"] == "s3":
        return DataHandler(bucket=config["s3_bucket"], storage_mode="s3")
    else:
        return DataHandler(storage_mode="local")

def main(sentiment_threshold=0.35):
    config = load_config()
    logger.info(f"[MLTrainingMain] config={config}")

    data_handler = build_data_handler(config)
    ticker = config["ticker"]
    date_range = f"{config['start_date']}_to_{config['end_date']}"

    # 1) Load Embeddings & Preprocessed Data
    embedding_filename = data_handler.construct_filename(ticker, "embeddings", date_range, "npy")
    preprocessed_filename = data_handler.construct_filename(ticker, "preprocessed", date_range, "csv")

    X = data_handler.load_data(embedding_filename, data_type="embeddings", stage="embeddings")
    df = data_handler.load_data(preprocessed_filename, data_type="csv", stage="preprocessed")

    if X is None or df is None or len(X) == 0 or df.empty:
        logger.error("Missing or empty embeddings / preprocessed data. Check previous stages.")
        return
    logger.info(f"Loaded embeddings shape={X.shape}, df shape={df.shape}")

    # Convert embeddings to a DataFrame
    column_names = [f"embedding_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=column_names, index=df.index)

    # 2) Initialize horizon manager, model manager, pipeline
    horizon_manager = TimeHorizonManager()
    combos = horizon_manager.generate_horizon_combos()
    combos = horizon_manager.filter_combos(combos, max_combos=2000)
    logger.info(f"Time horizon combos count={len(combos)}")

    model_manager = ModelManager(
        input_size=X_df.shape[1],
        hidden_layers=[256, 128, 64],
        learning_rate=0.001,
        batch_size=32,
        epochs=10,
        data_handler=data_handler,
        use_time_split=True
    )
    pipeline = ModelPipeline(model_manager, data_handler, horizon_manager)

    # 3) Example direct training across horizons (without Optuna):
    #    We'll do advanced filtering on multiple columns
    sentiment_cols = ["title_positive", "summary_positive", "expected_sentiment", "summary_negative"]
    pipeline.train_on_horizons(
        X_df,
        df,
        max_combos=2000,
        save_best_only=True,
        filter_sentiment=True,
        sentiment_threshold=sentiment_threshold,
        sentiment_cols=sentiment_cols,
        sentiment_mode="any"
    )
    logger.info("Finished direct horizon training pipeline.")

    # 4) (Optional) Now we show how to do Optuna with the same pipeline logic.
    #    We'll have each trial pick exactly 1 horizon from combos and run "train_on_horizons" for that horizon only.
    #    We'll parse the resulting MSE to return as the objective.

    # We define a helper that trains exactly one horizon:
    def train_single_horizon(combo_dict, X_df, df):
        """
        Reuses the pipeline but trains only the single horizon combo specified.
        Returns test MSE.
        """
        # We'll create a small sub-list of combos just with [combo_dict]
        single_combo = [combo_dict]

        # Call the pipeline with max_combos=1. We'll also pass the same sentiment filtering if desired.
        # But we want to retrieve the MSE from the pipeline results.
        results = pipeline.train_on_horizons(
            X_df,
            df,
            max_combos=1,  # We'll override the pipeline combos to just the single one
            save_best_only=True,
            filter_sentiment=True,
            sentiment_threshold=sentiment_threshold,
            sentiment_cols=sentiment_cols,
            sentiment_mode="any"
        )
        if not results:
            return float("inf")

        # results is a list of dicts with "mse", "mae", "r2"
        return results[0]["mse"]

    # Now define our Optuna objective:
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

        # Our target col might be e.g. "15_minutes_change" or "15_minutes_percentage_change"
        # We'll just do "_change" for now, so we check if it exists:
        target_col = f"{combo_dict['predict_name']}_change"
        if target_col not in df.columns:
            logger.warning(f"Missing target {target_col}, skipping combo.")
            return float("inf")

        # Train exactly this horizon using pipeline
        mse_val = train_single_horizon(combo_dict, X_df, df)
        return mse_val

    logger.info("Starting hyperparameter optimization with Optuna...")

    study = optuna.create_study(direction="minimize")

    def run_study_with_progress(study, n_trials):
        with tqdm(total=n_trials, desc="Optuna Trials") as pbar:
            def callback(study, trial):
                pbar.update(1)
            study.optimize(optimize, n_trials=n_trials, callbacks=[callback])

    run_study_with_progress(study, n_trials=25)
    logger.info(f"Best trial: {study.best_trial.number} with val={study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_trial.params}")

if __name__ == "__main__":
    # For demonstration, we pass sentiment_threshold=0.35 or so
    main(sentiment_threshold=0.35)
