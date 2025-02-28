import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import numpy as np
import pandas as pd
from .stock_predictor import StockPredictor
from .model_analysis import ModelAnalysis
from src.utils.logger import get_logger
from typing import Tuple, Any, List, Optional
from src.config import settings

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        """
        Early stopping utility to halt training if validation loss does not improve.
        
        :param patience: Number of epochs to wait.
        :param min_delta: Minimum improvement required to reset patience.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_loss = np.inf
        self.should_stop = False

    def __call__(self, val_loss: float) -> None:
        if val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

class ModelManager:
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = None,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        data_handler: Any = None,
        model_stage: str = "models",
        use_time_split: bool = True
    ) -> None:
        """
        Initializes the ModelManager.
        
        :param input_size: Number of input features.
        :param hidden_layers: List of hidden layer sizes.
        :param learning_rate: Learning rate for optimization.
        :param batch_size: Batch size.
        :param epochs: Number of training epochs.
        :param data_handler: DataHandler instance.
        :param model_stage: Directory/stage for model artifacts.
        :param use_time_split: Whether to split data using time-based criteria.
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers or [256, 128, 64]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_handler = data_handler
        self.model_stage = model_stage
        self.use_time_split = use_time_split
        self.logger = get_logger(self.__class__.__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, model_name: str, trial=None) -> Tuple[Any, dict]:
        """
        Trains the model and evaluates it.
        
        :param X: Features array.
        :param y: Target array.
        :param model_name: Name identifier for the model.
        :param trial: Optional Optuna trial for hyperparameter tuning.
        :return: Tuple of the trained model and a dictionary of metrics.
        """
        if self.use_time_split:
            if hasattr(X, "index"):
                X = X.sort_index()
                y = y[X.index]
            train_size = int(len(X) * 0.7)
            val_size = int(len(X) * 0.85)
            X_train, X_val, X_test = X[:train_size], X[train_size:val_size], X[val_size:]
            y_train, y_val, y_test = y[:train_size], y[train_size:val_size], y[val_size:]
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        if trial:
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
            n_layers = trial.suggest_int("n_layers", 1, 3)
            hidden_layers = [trial.suggest_int(f"n_units_l{i}", 32, 512, step=32) for i in range(n_layers)]
            batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
            dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        else:
            learning_rate = self.learning_rate
            hidden_layers = self.hidden_layers
            batch_size = self.batch_size
            dropout_rate = settings.model_dropout_rate

        train_loader = self._get_dataloader(X_train, y_train, batch_size)
        val_loader = self._get_dataloader(X_val, y_val, batch_size)

        model = StockPredictor(self.input_size, hidden_layers, dropout_rate=dropout_rate).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=settings.model_weight_decay)
        loss_fn = nn.SmoothL1Loss() if settings.model_loss_function.lower() == "smoothl1" else nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=settings.lr_scheduler_factor, patience=settings.lr_scheduler_patience
        )

        training_history, best_model_state, best_val_mse = self._train_model(
            model, train_loader, val_loader, optimizer, loss_fn, scheduler, model_name
        )

        if trial:
            return best_val_mse

        model.load_state_dict(best_model_state)
        y_test_pred = self._predict(model, X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        evs = explained_variance_score(y_test, y_test_pred)
        eps = 1e-8
        denom = np.where(np.abs(y_test) < eps, eps, y_test)
        mape = np.mean(np.abs((y_test - y_test_pred) / denom)) * 100
        tol = getattr(settings, "regression_accuracy_tolerance", 0.1)
        relative_errors = np.abs((y_test - y_test_pred) / denom)
        regression_accuracy = np.mean(relative_errors < tol) * 100
        slope, intercept = np.polyfit(y_test, y_test_pred, 1)
        line_of_best_fit_error = abs(slope - 1) + abs(intercept)

        from .model_summary import ModelSummary
        temp_summary = ModelSummary(self.data_handler)
        directional_accuracy = temp_summary.calculate_directional_accuracy(y_test_pred, y_test)
        percentage_over_prediction = temp_summary.calculate_percentage_over_prediction(y_test_pred, y_test)
        pearson_corr, spearman_corr = temp_summary.calculate_pearson_spearman(y_test_pred, y_test)

        from .model_analysis import ModelAnalysis
        analysis = ModelAnalysis(self.data_handler, model_stage=self.model_stage)
        analysis.generate_all_plots(
            training_history=training_history,
            y_true=y_test,
            y_pred=y_test_pred,
            model=model,
            X_np=X_test,
            model_name=model_name,
            value=test_mse
        )

        self.logger.info(
            f"Final metrics for '{model_name}': MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, "
            f"R²: {test_r2:.4f}, Explained Variance: {evs:.4f}, MAPE: {mape:.2f}%, "
            f"Regression Accuracy: {regression_accuracy:.2f}%, Line Fit Error: {line_of_best_fit_error:.4f}"
        )

        metrics = {
            "mse": test_mse,
            "mae": test_mae,
            "r2": test_r2,
            "explained_variance": evs,
            "mape": mape,
            "regression_accuracy": regression_accuracy,
            "line_of_best_fit_error": line_of_best_fit_error,
            "directional_accuracy": directional_accuracy,
            "percentage_over_prediction": percentage_over_prediction,
            "pearson_correlation": pearson_corr,
            "spearman_correlation": spearman_corr,
            "training_history": training_history,
        }
        return model, metrics

    def _train_model(self, model, train_loader, val_loader, optimizer, loss_fn, scheduler, model_name: str) -> Tuple[dict, dict, float]:
        """
        Trains the model over multiple epochs with early stopping.
        
        :param model: The neural network model.
        :param train_loader: DataLoader for training data.
        :param val_loader: DataLoader for validation data.
        :param optimizer: Optimizer.
        :param loss_fn: Loss function.
        :param scheduler: Learning rate scheduler.
        :param model_name: Identifier for logging.
        :return: Tuple of training history, best model state, and best validation MSE.
        """
        training_history = {"train_loss": [], "val_loss": []}
        early_stopper = EarlyStopping(patience=5, min_delta=0.0)
        best_model_state = None
        best_val_mse = float("inf")
        final_epoch = None

        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                y_pred = model(X_batch)
                loss = loss_fn(y_pred.view(-1), y_batch.view(-1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_train_loss = running_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            y_val_true = []
            y_val_pred = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    preds = model(X_batch)
                    loss = loss_fn(preds.view(-1), y_batch.view(-1))
                    val_loss += loss.item()
                    y_val_true.append(y_batch.cpu().numpy().reshape(-1))
                    y_val_pred.append(preds.cpu().numpy().reshape(-1))
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            y_val_true = np.concatenate(y_val_true)
            y_val_pred = np.concatenate(y_val_pred)
            val_mse = np.mean((y_val_true - y_val_pred) ** 2)

            training_history["train_loss"].append(avg_train_loss)
            training_history["val_loss"].append(avg_val_loss)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_model_state = model.state_dict()
            early_stopper(avg_val_loss)
            final_epoch = epoch + 1
            if early_stopper.should_stop:
                break

        self.logger.debug(f"Final Epoch {final_epoch}/{self.epochs} for '{model_name}' -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MSE: {val_mse:.4f}")
        return training_history, best_model_state, best_val_mse

    def _predict(self, model, X: np.ndarray) -> np.ndarray:
        model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = model(X_t)
        return preds.view(-1).cpu().numpy()

    def _get_dataloader(self, X: np.ndarray, y: np.ndarray, batch_size: int = None) -> DataLoader:
        from torch.utils.data import DataLoader, TensorDataset
        batch_size = batch_size or self.batch_size
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def save_model(self, model: nn.Module, filepath: str) -> None:
        """
        Saves the model state dictionary to the given filepath.
        
        :param model: Neural network model.
        :param filepath: Path to save the model.
        """
        torch.save(model.state_dict(), filepath)
