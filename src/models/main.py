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

    config = {
        "ticker": os.getenv("TICKER", "NVDA"),
        "start_date": os.getenv("START_DATE", "2022-01-01"),
        "end_date": os.getenv("END_DATE", "2024-01-31"),
        "local_mode": local_mode,
        "storage_mode": storage_mode,
        "s3_bucket": s3_bucket,
    }
    return config

def build_data_handler(config):
    if config["storage_mode"] == "s3":
        return DataHandler(bucket=config["s3_bucket"], storage_mode="s3")
    else:
        return DataHandler(storage_mode="local")

def main(sentiment_threshold):
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

    # Convert to a DataFrame with named columns
    column_names = [f"embedding_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=column_names, index=df.index)

    # 2) Initialize Time Horizons, ModelManager, and Pipeline
    horizon_manager = TimeHorizonManager()
    combos = horizon_manager.generate_horizon_combos()
    combos = horizon_manager.filter_combos(combos, max_combos=2000)  # optional
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

    # 3) (Optional) Basic Trials or direct training
    #    Example direct training across horizons, with optional sentiment filtering:
    results = pipeline.train_on_horizons(
        X_df, 
        df, 
        max_combos=1000,            # Train on up to 10 horizon combos
        save_best_only=True,      # Only save best model for each horizon
        filter_sentiment=True,    # Remove rows below threshold
        sentiment_threshold=0.2   # You can tune this threshold
    )
    logger.info(f"Training complete. Results: {results}")

    # 4) (Optional) Or run an Optuna optimization:
    def baseline_mse(y_true):
        return mean_squared_error(y_true, [y_true.mean()] * len(y_true))

    # We'll do a small example with Optuna if desired
    def optimize(trial):
        import json
        selected_combo = trial.suggest_categorical("horizon_combo", [
            json.dumps({
                "gather_name": c["gather_name"],
                "gather_td": c["gather_td"].total_seconds(),
                "predict_name": c["predict_name"],
                "predict_td": c["predict_td"].total_seconds()
            }) for c in combos
        ])
        combo = json.loads(selected_combo)
        target_col = f"{combo['predict_name']}_percentage_change"
        if target_col not in df.columns:
            logger.warning(f"Missing target {target_col}, skipping.")
            return float("inf")

        df_filtered = df.dropna(subset=[target_col]).copy()
        # If filtering sentiment in the trial, do it here:
        df_filtered = pipeline.filter_low_impact_sentiment(df_filtered, threshold=sentiment_threshold)
        X_f = X_df.loc[df_filtered.index]
        y_f = df_filtered[target_col].values
        if X_f.empty:
            return float("inf")

        val_mse = model_manager.train_and_evaluate(X_f.values, y_f, model_name=f"trial_{trial.number}", trial=trial)
        return val_mse

    # Uncomment to actually run Optuna
    # import optuna
    # from tqdm import tqdm

    # study = optuna.create_study(direction="minimize")
    # logger.info("Starting hyperparameter optimization with Optuna...")

    # def run_study_with_progress(study, n_trials):
    #     with tqdm(total=n_trials, desc="Optuna Trials") as pbar:
    #         def callback(study, trial):
    #             pbar.update(1)
    #         study.optimize(optimize, n_trials=n_trials, callbacks=[callback])

    # run_study_with_progress(study, n_trials=50)
    # logger.info(f"Best trial: {study.best_trial.number} val={study.best_trial.value}")
    # logger.info(f"Best params: {study.best_trial.params}")


if __name__ == "__main__":
    main(sentiment_threshold = 0.4)
