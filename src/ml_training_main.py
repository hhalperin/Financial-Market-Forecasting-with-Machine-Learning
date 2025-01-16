# ml_training_main.py

import os
import time
import numpy as np
import pandas as pd
from datetime import timedelta
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.data_handler import DataHandler
from models.model_manager import ModelManager
from data_processing.time_horizon_manager import TimeHorizonManager
from sklearn.model_selection import train_test_split

logger = get_logger("MLTrainingMain")

def compute_expected_sentiment_for_horizon(df, gather_horizon):
    """
    gather_horizon: e.g. '30_minutes' => weigh sentiment by '30_minutes_change'
    """
    from data_processing.expected_sentiment_calculator import ExpectedSentimentCalculator

    reaction_col = f"{gather_horizon}_change"
    if reaction_col not in df.columns:
        logger.warning(f"{reaction_col} not found, skipping expected_sentiment.")
        df['expected_sentiment'] = 0
        return df

    esc = ExpectedSentimentCalculator(
        data_df=df,
        window_size=5,
        weight_by_reaction=True
    )
    df = esc.compute_expected_sentiment(
        sentiment_col='summary_sentiment',
        reaction_col=reaction_col
    )
    return df

def main():
    start_time = time.time()
    load_dotenv()

    local_mode_str = os.getenv("LOCAL_MODE", "true").lower()
    local_mode = (local_mode_str == 'true')
    storage_mode = 'local' if local_mode else 's3'
    logger.info(f"[MLTrainingMain] local_mode={local_mode}, storage_mode={storage_mode}")

    # Build DataHandler
    if storage_mode == 'local':
        data_handler = DataHandler(base_data_dir='../data', storage_mode='local')
    else:
        s3_bucket = os.getenv("S3_BUCKET", None)
        data_handler = DataHandler(bucket=s3_bucket, base_data_dir='../data', storage_mode='s3')

    # 1) Load embeddings + preprocessed DataFrame
    ticker = "AAPL"
    date_range = "2023-01-01_to_2024-01-31"

    def fetch_no_op_embeddings():
        return np.array([])

    X = data_handler(ticker, date_range, "embeddings", fetch_no_op_embeddings, stage='embeddings')
    if X is None or X.size == 0:
        logger.error("No embeddings found. Please run embedding step first.")
        return
    logger.info(f"Embeddings shape: {X.shape}")

    def fetch_no_op_df():
        return pd.DataFrame()

    df = data_handler(ticker, date_range, "preprocessed", fetch_no_op_df, stage='preprocessed')

    if df is None or df.empty:
        logger.error("No preprocessed DataFrame found. Please run data_processing first.")
        return
    logger.info(f"Preprocessed DF shape: {df.shape}")

    # 2) Generate many horizon combos dynamically
    thm = TimeHorizonManager()
    combos = thm.generate_horizon_combos(max_combos=100)  # Generate 100 combos
    if not combos:
        logger.warning("No horizon combos generated.")
        return
    logger.info(f"Generated {len(combos)} horizon combos.")

    results = []  # Initialize results list for model comparisons

    # 3) For each combo, compute expected_sentiment if desired, then pick the predict horizon as the target
    from tqdm import tqdm
    for combo in tqdm(combos, desc="Horizon Combos"):
        gather_horizon = combo['gather_name']
        predict_horizon = combo['predict_name']
        gather_col = f"{gather_horizon}_change"
        predict_col = f"{predict_horizon}_change"

        logger.info(f"--- GATHER: {gather_col}, PREDICT: {predict_col} ---")

        df_horizon = df.copy()
        df_horizon = compute_expected_sentiment_for_horizon(df_horizon, gather_horizon)
        df_horizon = df_horizon.dropna(subset=[gather_col, predict_col])

        X_arr = X
        if 'expected_sentiment' in df_horizon.columns:
            extra_feat = df_horizon['expected_sentiment'].values.reshape(-1,1)
            import numpy as np
            X_arr = np.hstack([X_arr, extra_feat])

        y_arr = df_horizon[predict_col].values
        if len(y_arr) != X_arr.shape[0]:
            logger.warning(f"Mismatch shapes: X_arr={X_arr.shape[0]}, y_arr={len(y_arr)}. Skipping combo.")
            continue

        from sklearn.model_selection import train_test_split
        X_train_val, X_test, y_train_val, y_test = train_test_split(X_arr, y_arr, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.111, random_state=42)

        from models.model_manager import ModelManager
        input_size = X_arr.shape[1]
        hidden_layers = [256, 128, 64]
        model_manager = ModelManager(
            input_size=input_size,
            hidden_layers=hidden_layers,
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            data_handler=data_handler,
            model_stage='models'
        )

        model_name = f"model_{gather_horizon}_to_{predict_horizon}"
        logger.info(f"--- Training model: {model_name} ---")
        trained_model, mse, r2 = model_manager.train_and_evaluate(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            timestamps=None,
            model_name=model_name
        )
        logger.info(f"Finished combo {gather_horizon} -> {predict_horizon}: MSE={mse:.4f}, R2={r2:.4f}")

        results.append({
            "gather": gather_horizon,
            "predict": predict_horizon,
            "model_name": model_name,
            "mse": mse,
            "r2": r2,
            "hyperparameters": {
                "hidden_layers": hidden_layers,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10
            }
        })

    # After loop: find best model
    if results:
        best_overall = min(results, key=lambda x: x["mse"])
        logger.info("=== Best Overall Model ===")
        logger.info(f"Time Horizon: Gather {best_overall['gather']} -> Predict {best_overall['predict']}")
        logger.info(f"Model Name: {best_overall['model_name']}")
        logger.info(f"Best Hyperparameters: {best_overall['hyperparameters']}")
        logger.info(f"Performance: MSE={best_overall['mse']:.4f}, R2={best_overall['r2']:.4f}")
    else:
        logger.warning("No results to compare.")

    elapsed = time.time() - start_time
    logger.info(f"ML training completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
