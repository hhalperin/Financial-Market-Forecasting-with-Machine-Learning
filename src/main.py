import time
import pandas as pd
from data_aggregation import DataAggregator
from data_processing.preprocessing_manager import PreprocessingManager, TimeHorizonManager
from models.model_pipeline import ModelPipeline
from sklearn.model_selection import train_test_split
from models import ModelManager
from logger import get_logger

def main():
    logger = get_logger('Main')
    use_batch_api = False  # Set this to True to use batch API submission/retrieval

    total_start_time = time.time()
    logger.info("Starting data aggregation...")
    # Step 1: Aggregate Data
    try:
        aggregator = DataAggregator(ticker='AAPL', start_date='2021-01-01', end_date='2021-06-30')
        price_df, news_df = aggregator.aggregate_data()
        
        if price_df is None or price_df.empty:
            logger.error("Price data aggregation resulted in an empty or None DataFrame.")
            return
        if news_df is None or news_df.empty:
            logger.error("News data aggregation resulted in an empty or None DataFrame.")
            return
    except Exception as e:
        logger.error(f"Error during data aggregation: {e}")
        return

    logger.info("Data aggregation completed. Starting preprocessing...")
    # Step 2: Preprocess Data
    preprocessing_manager = PreprocessingManager(pd.concat([price_df, news_df], axis=1))
    
    # Step 2.1: Generate time horizons for dynamic target calculation
    time_horizon_manager = TimeHorizonManager(start_date, end_date)
    time_horizons = time_horizon_manager.generate_time_horizons()
    if not time_horizons:
        logger.error("No valid time horizons generated. Exiting.")
        return

    try:
        # Perform the entire preprocessing pipeline before filtering
        preprocessing_manager.preprocess(target_configs=[], time_horizons=time_horizons)
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return

    # Step 3: Filter on article release events
    filtered_df = preprocessing_manager.filter_on_article_release()
    if filtered_df is None or filtered_df.empty:
        logger.error("Filtered DataFrame is empty after article release filtering.")
        return

    # Step 4: Calculate dynamic targets
    preprocessing_manager.df = filtered_df
    try:
        preprocessing_manager.calculate_dynamic_targets(column_name='Close', target_configs=time_horizons)
    except Exception as e:
        logger.error(f"Error during dynamic target calculation: {e}")
        return

    # Step 5: Handle embeddings
    logger.info("Generating or retrieving embeddings...")
    pipeline = ModelPipeline(ticker, start_date, end_date, use_batch_api=use_batch_api)
    try:
        embeddings = pipeline.handle_embeddings(preprocessing_manager.df, config_id="default_config")
    except Exception as e:
        logger.error(f"Error during embeddings generation: {e}")
        return

    if embeddings is None:
        logger.error("Failed to generate or retrieve embeddings.")
        return

    # Ensure that the number of embeddings matches the original data
    X = embeddings
    if X.shape[0] != len(preprocessing_manager.df):
        logger.error("Mismatch between number of embeddings and data.")
        return

    # Step 6: Extract target variable and timestamps for model training
    logger.info("Preparing data for model training and evaluation...")
    target_column = '7_days_change'  # Assuming a specific target column is selected after preprocessing
    if target_column not in preprocessing_manager.df.columns:
        logger.error(f"Target column '{target_column}' not found in preprocessed data.")
        return

    y = preprocessing_manager.df[target_column].fillna(0).values
    timestamps = preprocessing_manager.df['time_published'] if 'time_published' in preprocessing_manager.df.columns else range(len(preprocessing_manager.df))

    # Step 7: Split the data into training, validation, and test sets
    logger.info("Splitting data into training, validation, and test sets...")
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    except Exception as e:
        logger.error(f"Error during train/test split: {e}")
        return

    # Step 8: Train and evaluate models
    logger.info("Starting model training and evaluation...")
    try:
        input_size = X_train.shape[1]
        hidden_layers = [256, 128, 64]  # Initial values for hidden layers

        model_manager = ModelManager(input_size, hidden_layers)

        # Step 8.1: Pass the model manager to the pipeline to train and evaluate models for different horizons
        pipeline.train_and_evaluate_models(X_train, filtered_df, time_horizons, model_manager)
    except Exception as e:
        logger.error(f"Error during model training and evaluation: {e}")
        return

    # Step 9: Run hyperparameter optimization via ModelManager
    logger.info("Starting hyperparameter optimization using Optuna...")
    try:
        model_manager.optimize_with_optuna(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            input_size=input_size
        )
    except Exception as e:
        logger.error(f"Error during hyperparameter optimization: {e}")
        return

    # Log total execution time
    total_end_time = time.time()
    logger.info(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
