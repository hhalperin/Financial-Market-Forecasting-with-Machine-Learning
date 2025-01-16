# best_model_training_main.py

import os
import torch
import pandas as pd
from models.stock_predictor import StockPredictor
from src.utils.logger import get_logger
from data_aggregation import aggregate_data
from data_processing import preprocess_data, TimeHorizonManager
from models.model_pipeline import ModelPipeline
from sklearn.model_selection import train_test_split

def load_best_model(model_path, input_size, hidden_layers):
    """
    Load the best model from the specified path.
    
    Args:
        model_path (str): Path to the saved best model.
        input_size (int): The input size of the model.
        hidden_layers (list of int): The number of units in each hidden layer.

    Returns:
        nn.Module: Loaded model ready for evaluation.
    """
    # Instantiate the model architecture
    model = StockPredictor(input_size, hidden_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

def main():
    logger = get_logger('BestModelTesting')
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01'  # Test on a larger date range

    # Aggregate Data
    price_df, news_df = aggregate_data(ticker, start_date, end_date)

    # Preprocess Data
    preprocessed_df = preprocess_data(ticker, start_date, end_date, price_df, news_df)

    # Handle embeddings
    pipeline = ModelPipeline(ticker, start_date, end_date, use_batch_api=False)
    embeddings = pipeline.handle_embeddings(preprocessed_df, config_id="best_model_test")
    if embeddings is None:
        logger.error("Failed to generate or retrieve embeddings.")
        return

    # Extract features and split data
    X = embeddings
    assert X.shape[0] == len(preprocessed_df), "Mismatch between number of embeddings and data"
    y = preprocessed_df['Close'].fillna(0).values  # Assuming 'Close' is the target

    # Split data for testing purposes
    logger.info("Splitting data into training and test sets...")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # We only need a test set

    # Define the path to the best model
    model_dir = 'saved_models/best'
    model_path = os.path.join(model_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        logger.error(f"Best model not found at {model_path}")
        return

    # Load the best model
    input_size = X_test.shape[1]
    hidden_layers = [256, 128, 64]  # Assuming this matches the saved model configuration
    best_model = load_best_model(model_path, input_size, hidden_layers)

    # Evaluate the model on the test set
    best_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        predictions = best_model(X_test_tensor).squeeze()

    # Calculate and log performance metrics
    mse = torch.nn.functional.mse_loss(predictions, y_test_tensor).item()
    logger.info(f'Test MSE for best model: {mse:.4f}')

if __name__ == '__main__':
    main()
