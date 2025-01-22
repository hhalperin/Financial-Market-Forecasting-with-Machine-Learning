import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from .stock_predictor import StockPredictor
from .model_analysis import ModelAnalysis
from src.utils.logger import get_logger

class EarlyStopping:
    """
    Simple EarlyStopping mechanism: if validation loss doesn't improve for 'patience' epochs, stop training.
    """
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_loss = np.inf
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

class ModelManager:
    """
    Manages creation, training, and evaluation of our PyTorch model for
    predicting stock changes from embeddings or other features.
    """

    def __init__(
        self,
        input_size,
        hidden_layers=None,
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        data_handler=None,
        model_stage="models",   # <= By default, uses the "models" stage => data/models on disk
        use_time_split=True
    ):
        """
        :param input_size: Number of input features.
        :param hidden_layers: e.g. [256, 128, 64].
        :param learning_rate: Optimizer LR.
        :param batch_size: Training batch size.
        :param epochs: Max epochs.
        :param data_handler: For saving data & figures.
        :param model_stage: The 'stage' name. data_handler merges it into data/<stage>.
        :param use_time_split: If True, do a chronological train/val/test split.
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

        # Detect GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

    def train_and_evaluate(self, X, y, model_name, trial=None):
        """
        Main entry point: trains a neural network on X->y, returns metrics or val MSE if in an Optuna trial.

        :param X: Numpy array shape (n_samples, n_features).
        :param y: Numpy array shape (n_samples,).
        :param model_name: A name for logging & saving artifacts.
        :param trial: Optional. If provided, returns val MSE for Optuna.
        :return: (model, metrics) or val MSE if trial is given.
        """

        # 1. Split data
        if self.use_time_split:
            # If X is a DataFrame, sort by index
            if hasattr(X, "index"):
                X = X.sort_index()
                y = y[X.index]

            train_size = int(len(X) * 0.7)
            val_size = int(len(X) * 0.85)
            X_train, X_val, X_test = X[:train_size], X[train_size:val_size], X[val_size:]
            y_train, y_val, y_test = y[:train_size], y[train_size:val_size], y[val_size:]
        else:
            # random split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # 2. Possibly sample hyperparams from trial
        if trial:
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
            n_layers = trial.suggest_int("n_layers", 1, 3)
            hidden_layers = []
            for i in range(n_layers):
                hidden_layers.append(trial.suggest_int(f"n_units_l{i}", 32, 512, step=32))
            batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
        else:
            learning_rate = self.learning_rate
            hidden_layers = self.hidden_layers
            batch_size = self.batch_size

        # 3. Create data loaders
        train_loader = self._get_dataloader(X_train, y_train, batch_size)
        val_loader = self._get_dataloader(X_val, y_val, batch_size)

        # 4. Initialize model
        model = StockPredictor(self.input_size, hidden_layers).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()

        # 5. Train model (early stopping)
        training_history, best_model_state, best_val_mse = self._train_model(
            model, train_loader, val_loader, optimizer, loss_fn
        )

        # 6. If we're in an Optuna trial, just return the best val MSE
        if trial:
            return best_val_mse

        # 7. Evaluate on test set
        model.load_state_dict(best_model_state)
        y_test_pred = self._predict(model, X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # 8. Generate plots
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

        # 9. Compare to baseline
        baseline_pred = np.full_like(y_test, y_test.mean())
        baseline_val = mean_squared_error(y_test, baseline_pred)
        self.logger.info(f"Test MSE: {test_mse:.4f}, Baseline MSE: {baseline_val:.4f}")

        # 10. Return model & metrics
        metrics = {
            "mse": test_mse,
            "mae": test_mae,
            "r2": test_r2,
            "training_history": training_history,
        }
        return model, metrics

    def _train_model(self, model, train_loader, val_loader, optimizer, loss_fn):
        """
        Core training loop with early stopping, returning:
          - training_history
          - best_model_state
          - best_val_mse
        """
        training_history = {"train_loss": [], "val_loss": []}
        early_stopper = EarlyStopping(patience=5, min_delta=0.0)

        best_model_state = None
        best_val_mse = float("inf")

        for epoch in range(self.epochs):
            # -- Training --
            model.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = loss_fn(y_pred.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)

            # -- Validation --
            model.eval()
            val_loss = 0.0
            y_val_true = []
            y_val_pred = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    preds = model(X_batch)
                    loss = loss_fn(preds.squeeze(), y_batch)
                    val_loss += loss.item()

                    y_val_true.append(y_batch.cpu().numpy())
                    y_val_pred.append(preds.squeeze().cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)

            y_val_true = np.concatenate(y_val_true)
            y_val_pred = np.concatenate(y_val_pred)
            val_mse = mean_squared_error(y_val_true, y_val_pred)

            training_history["train_loss"].append(avg_train_loss)
            training_history["val_loss"].append(avg_val_loss)
            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MSE: {val_mse:.4f}"
            )

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_model_state = model.state_dict()

            early_stopper(val_mse)
            if early_stopper.should_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        return training_history, best_model_state, best_val_mse

    def _predict(self, model, X):
        """
        Predict with the given model, returning NumPy.
        """
        model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = model(X_t)
        return preds.squeeze().cpu().numpy()

    def _get_dataloader(self, X, y, batch_size=None):
        """
        Returns a DataLoader for (X, y).
        """
        batch_size = batch_size or self.batch_size
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def save_model(self, model, filepath):
        """
        Saves the .pt file. 
        """
        torch.save(model.state_dict(), filepath)
        self.logger.info(f"Model saved to {filepath}.")
