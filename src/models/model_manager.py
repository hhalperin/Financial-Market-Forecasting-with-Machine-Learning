# models/model_manager.py

import torch
from .stock_predictor import StockPredictor
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator
from .model_visualizer import ModelVisualizer
from utils.logger import get_logger

class ModelManager:
    """
    Coordinates model creation, training, evaluation, and optional Optuna hyperparam search.
    """

    def __init__(self, input_size, hidden_layers, learning_rate=0.001, batch_size=32,
                 epochs=50, data_handler=None, model_stage='models'):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_handler = data_handler
        self.model_stage = model_stage
        self.logger = get_logger(self.__class__.__name__)

    def _train_and_evaluate_model(self, X_train, y_train, X_val, y_val,
                                  X_test, y_test,
                                  model_name="best_model"):
        # 1) Build the model
        model = StockPredictor(self.input_size, self.hidden_layers)

        # 2) Trainer
        trainer = ModelTrainer(
            model,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epochs=self.epochs,
            data_handler=self.data_handler,
            model_stage=self.model_stage
        )
        trained_model, training_history = trainer.train(X_train, y_train, X_val, y_val, model_name=model_name)

        # 3) Evaluate
        evaluator = ModelEvaluator(trained_model, data_handler=self.data_handler, model_stage=self.model_stage)
        mse, mae, r2 = evaluator.evaluate(X_test, y_test, model_name=model_name)

        # 4) Visualize
        visualizer = ModelVisualizer(self.data_handler, model_stage=self.model_stage, model_name=model_name)
        visualizer.plot_learning_curve(training_history)
        # Plot actual vs predicted
        import torch
        preds = trained_model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
        visualizer.plot_actual_vs_predicted(y_test, preds)

        return trained_model, mse, r2

    def train_and_evaluate(self, X_train, y_train, X_val, y_val,
                           X_test, y_test, timestamps=None,
                           model_name="best_model"):
        # train & evaluate
        trained_model, mse, r2 = self._train_and_evaluate_model(
            X_train, y_train, X_val, y_val,
            X_test, y_test,
            model_name=model_name
        )
        # optional time series plot
        if timestamps is not None and len(timestamps) == len(y_test):
            visualizer = ModelVisualizer(self.data_handler, model_stage=self.model_stage, model_name=model_name)
            import torch
            preds = trained_model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
            visualizer.plot_time_series(timestamps, y_test, preds)

        return trained_model, mse, r2

    @staticmethod
    def optimize_with_optuna(X_train, y_train, X_val, y_val, X_test, y_test,
                             input_size, n_trials=50, data_handler=None):
        import optuna
        logger = get_logger('OptunaOptimization')

        def objective(trial):
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            epochs = trial.suggest_int('epochs', 3, 10)
            hidden_layers = trial.suggest_categorical('hidden_layers', [[256,128,64], [512,256], [128,64]])

            # For each trial, build a manager & train
            manager = ModelManager(input_size, hidden_layers, learning_rate, batch_size,
                                   epochs, data_handler=data_handler)
            # We'll do a simple train/val split
            trained_model, mse, r2 = manager._train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test)
            return mse  # minimize MSE

        logger.info("Starting hyperparameter optimization with Optuna...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        logger.info(f"Best hyperparameters found: {study.best_params}")
        return study
