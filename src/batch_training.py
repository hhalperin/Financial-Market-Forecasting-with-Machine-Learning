import os
import json
import boto3
import numpy as np
import pandas as pd
from src.utils.logger import get_logger
from src.utils.data_handler import DataHandler
from models.model_trainer import train_model

logger = get_logger("BatchTraining")

if __name__ == "__main__":
    # This script is meant to be run on AWS Batch
    # It expects environment variables or a JSON file passed as input
    
    # In AWS Batch, you might pass in parameters via environment variables or a file
    input_json_path = os.environ.get('JOB_INPUT_JSON', '/tmp/input.json')
    with open(input_json_path, 'r') as f:
        event = json.load(f)

    ticker = event['ticker']
    start_date = event['start_date']
    end_date = event['end_date']
    embeddings_key = event['embeddings_key']
    time_horizons = event['time_horizons']  # List of horizons

    s3_bucket = os.environ['S3_BUCKET']
    data_handler = DataHandler(s3_bucket)

    logger.info("Starting training job...")
    logger.info(f"Fetching embeddings: {embeddings_key}")

    embeddings = data_handler.load_data(embeddings_key, data_type='embeddings')
    preprocessed_key = f"{ticker}_preprocessed_{start_date}_to_{end_date}.csv"
    df = data_handler.load_data(preprocessed_key, data_type='preprocessed')

    if embeddings is None or len(embeddings) == 0:
        logger.error("No embeddings loaded.")
        exit(1)
    if df is None or df.empty:
        logger.error("No preprocessed data.")
        exit(1)

    # For demonstration, pick a target column from time_horizons
    target_config = time_horizons[0]
    target_name = target_config['target_name']
    if target_name not in df.columns:
        logger.error(f"Target {target_name} not in DataFrame.")
        exit(1)

    y = df[target_name].values
    X = embeddings
    if X.shape[0] != y.shape[0]:
        logger.error("Embeddings and target size mismatch.")
        exit(1)

    # Simple train-test split
    split = int(0.8*len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model_artifact_key = f"{ticker}_model_{start_date}_to_{end_date}.pt"
    train_model(X_train, y_train, X_test, y_test, data_handler, model_artifact_key)

    logger.info("Training completed successfully.")
    output = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "model_artifact_key": model_artifact_key
    }

    # Save output to file, in Batch you can upload results to S3 or send events back via Step Functions callback
    with open('/tmp/output.json', 'w') as f:
        json.dump(output, f)
