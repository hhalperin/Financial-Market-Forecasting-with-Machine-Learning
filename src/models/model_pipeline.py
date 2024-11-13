import os
import json
import numpy as np
from datetime import timedelta
from data_processing import PreprocessingManager, TimeHorizonManager, DataEmbedder
from utils.api_batch import submit_batch_job, retrieve_batch_results
from logger import get_logger
from sklearn.model_selection import train_test_split
from utils.data_handler import DataHandler

class ModelPipeline:
    def __init__(self, ticker, start_date, end_date, use_batch_api=False):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.use_batch_api = use_batch_api
        self.logger = get_logger('ModelPipeline')
        self.embedded_dir = os.path.join('data', 'embeddings')
        os.makedirs(self.embedded_dir, exist_ok=True)
        self.data_handler = DataHandler()

    def handle_embeddings(self, preprocessed_df, config_id):
        embeddings_filepath = os.path.join(self.embedded_dir, f"{self.ticker}_{self.start_date}_to_{self.end_date}_{config_id}.npy")

        if self.use_batch_api and not os.path.exists(embeddings_filepath):
            batch_id = self._handle_batch_submission(preprocessed_df, config_id)
            if batch_id:
                embeddings = retrieve_batch_results(batch_id)
            else:
                self.logger.error("Batch ID was not found, cannot retrieve embeddings.")
                return None

        elif not self.use_batch_api:
            self.logger.info("Generating embeddings synchronously without batch API...")
            embedder = DataEmbedder(model_type='openai', model_name='text-embedding-3-small', n_components=512)
            embeddings = embedder.create_embeddings_from_dataframe(
                preprocessed_df, self.ticker, f"{self.start_date}_to_{self.end_date}", self.data_handler, config_id=config_id
            )
        else:
            self.logger.info(f"Loading existing embeddings from {embeddings_filepath}...")
            embeddings = np.load(embeddings_filepath)

        return embeddings

    def train_and_evaluate_models(self, X, preprocessed_df, time_horizons, model_manager):
        """
        Train and evaluate models for each prediction horizon.

        Args:
            X (np.ndarray): Feature matrix.
            preprocessed_df (pd.DataFrame): Preprocessed DataFrame to be used for training.
            time_horizons (List[Dict]): List of time horizons for training.
            model_manager (ModelManager): Instance of ModelManager to train and evaluate models.
        """
        for config in time_horizons:
            target_name = config['target_name']
            time_horizon = config['time_horizon']

            self.logger.info(f"Training model for time horizon: {time_horizon}")

            # Step 1: Initialize PreprocessingManager
            preprocessing_manager = PreprocessingManager(preprocessed_df)

            # Step 2: Filter data based on the event time horizon
            filtered_df = preprocessing_manager.filter_pre_event_data('time_published', time_horizon)

            # Step 3: Calculate dynamic targets
            preprocessing_manager.df = filtered_df  # Update manager with the filtered DataFrame
            preprocessing_manager.calculate_dynamic_targets(column_name='Close', target_configs=[config])

            # Step 4: Validate if the target column exists
            if target_name not in preprocessing_manager.df.columns:
                self.logger.error(f"Target column '{target_name}' not found in DataFrame.")
                continue

            # Step 5: Prepare the features and target variable
            y = preprocessing_manager.df[target_name].fillna(0).values

            # Split data into training, validation, and test sets
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)

            # Step 6: Train and evaluate the model
            self.logger.info(f"Training model for {target_name} prediction with default hyperparameters...")
            model_manager.train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)

    def optimize_hyperparameters(self, X, preprocessed_df, model_manager):
        """
        Run Optuna for hyperparameter optimization.

        Args:
            X (np.ndarray): Feature matrix.
            preprocessed_df (pd.DataFrame): Preprocessed DataFrame.
            model_manager (ModelManager): Instance of ModelManager to optimize hyperparameters.
        """
        # Use a target column with a clear goal for optimization
        self.logger.info("Starting hyperparameter optimization using Optuna...")

        # Step 1: Initialize PreprocessingManager
        preprocessing_manager = PreprocessingManager(preprocessed_df)

        # Step 2: Filter data for target calculation (assuming a specific time horizon)
        time_horizon = timedelta(days=7)
        filtered_df = preprocessing_manager.filter_pre_event_data('time_published', time_horizon)

        # Step 3: Calculate dynamic targets
        target_configs = [{'time_horizon': time_horizon, 'target_name': '7_days_change'}]
        preprocessing_manager.df = filtered_df  # Update manager with the filtered DataFrame
        preprocessing_manager.calculate_dynamic_targets(column_name='Close', target_configs=target_configs)

        # Step 4: Validate if the target exists
        if '7_days_change' not in preprocessing_manager.df.columns:
            self.logger.error("Target column '7_days_change' not found in DataFrame.")
            return

        # Step 5: Prepare the features and target variable
        y = preprocessing_manager.df['7_days_change'].fillna(0).values

        # Split data into training, validation, and test sets
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)

        # Step 6: Use Optuna to optimize hyperparameters
        input_size = X_train.shape[1]
        study = model_manager.optimize_with_optuna(X_train, y_train, X_val, y_val, X_test, y_test, input_size)
        self.logger.info(f"Best hyperparameters found by Optuna: {study.best_params}")

    def _split_data(self, X, y, test_size=0.3, validation_size=0.5, random_state=42):
        """
        Split the data into training, validation, and test sets.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            test_size (float): Proportion of the dataset to include in the test split.
            validation_size (float): Proportion of the dataset to include in the validation split from the test split.
            random_state (int): Random seed for reproducibility.

        Returns:
            Tuple of split data arrays (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        self.logger.info("Splitting data into training, validation, and test sets...")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _handle_batch_submission(self, preprocessed_df, config_id):
        # Submit a batch job and save the batch ID
        batch_id_filepath = os.path.join(self.embedded_dir, f"batch_id_{config_id}.json")

        if not os.path.exists(batch_id_filepath):
            batch_id = submit_batch_job(self.ticker, self.start_date, self.end_date, preprocessed_df)
            if batch_id:
                with open(batch_id_filepath, 'w') as f:
                    json.dump({'batch_id': batch_id}, f)
            return batch_id
        else:
            with open(batch_id_filepath, 'r') as f:
                return json.load(f).get('batch_id')
