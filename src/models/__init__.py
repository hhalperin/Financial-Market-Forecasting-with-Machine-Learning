import torch
import optuna
from .stock_predictor import StockPredictor
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator
from .model_visualizer import ModelVisualizer
from .model_pipeline import ModelPipeline
from logger import get_logger

class ModelManager:
    """
    Manages the end-to-end process of training, evaluating, and saving models.
    """

    def __init__(self, input_size, hidden_layers, learning_rate=0.001, batch_size=32, epochs=50):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.visualizer = ModelVisualizer()
        self.logger = get_logger(self.__class__.__name__)

    def _train_and_evaluate_model(self, X_train, y_train, X_val, y_val, X_test, y_test, model_name="best_model"):
        """
        Shared method for training and evaluating a model.

        Args:
            X_train, y_train, X_val, y_val, X_test, y_test: Data arrays for training, validation, and testing.
            model_name (str): The name for the model when saving results.

        Returns:
            Trained model and evaluation metrics.
        """
        # Instantiate the model and ModelTrainer
        model = StockPredictor(self.input_size, self.hidden_layers)
        trainer = ModelTrainer(model, learning_rate=self.learning_rate, batch_size=self.batch_size, epochs=self.epochs)

        # Train the model
        trained_model, training_history = trainer.train(X_train, y_train, X_val, y_val)

        # Log and visualize training history
        for train_loss, val_loss in zip(training_history['train_loss'], training_history['val_loss']):
            self.visualizer.log_metrics(train_loss, val_loss)

        # Plot learning curve
        self.visualizer.plot_learning_curve(model_name=model_name)

        # Predict on the test set to gather predictions
        y_pred = trained_model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()

        # Evaluate the Model
        evaluator = ModelEvaluator(trained_model)
        mse, r2 = evaluator.evaluate(X_test, y_test)
        self.logger.info(f"Test MSE: {mse:.4f}, R2 Score: {r2:.4f}")

        # Visualize additional evaluation metrics
        self.visualizer.plot_actual_vs_predicted(y_test, y_pred, model_name=model_name)

        return trained_model, mse, r2

    def train_and_evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test, timestamps):
        """
        Train the model and evaluate its performance.

        Args:
            X_train, y_train, X_val, y_val, X_test, y_test: Data arrays for training, validation, and testing.
            timestamps: List of timestamps associated with test data points.

        Returns:
            Trained model and evaluation metrics.
        """
        trained_model, mse, r2 = self._train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, model_name="best_model")

        # Ensure timestamps are aligned with y_test and y_pred, and pass them to the method
        if len(timestamps) == len(y_test):
            self.visualizer.plot_time_series(timestamps, y_test, trained_model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy(), model_name="best_model")
        else:
            self.logger.warning("Length mismatch between timestamps and y_test/y_pred. Skipping time series plot.")

        self.visualizer.plot_cumulative_returns(y_test, trained_model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy(), model_name="best_model")

        return trained_model, mse, r2

    @staticmethod
    def optimize_with_optuna(X_train, y_train, X_val, y_val, X_test, y_test, input_size, n_trials=50):
        """
        Use Optuna to optimize model hyperparameters.
        """
        def objective(trial):
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            epochs = trial.suggest_int('epochs', 3, 10)  # Reduce epochs during optimization to speed up
            hidden_layers = trial.suggest_categorical('hidden_layers', [[256, 128, 64], [512, 256], [128, 64]])

            model_manager = ModelManager(input_size, hidden_layers, learning_rate, batch_size, epochs)
            _, mse, _ = model_manager._train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, model_name="optuna_trial")

            return mse

        logger = get_logger('OptunaOptimization')
        logger.info("Starting hyperparameter optimization using Optuna...")

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Best hyperparameters found by Optuna: {study.best_params}")
        return study
