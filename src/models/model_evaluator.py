import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from utils.logger import get_logger
import logging

class ModelEvaluator:
    """
    Evaluates the trained model on test data.
    """

    def __init__(self, model):
        self.model = model
        self.logger = get_logger(self.__class__.__name__)

    def evaluate(self, X_test, y_test):
        """
        Evaluates model performance.
        """
        self.model.eval()
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=32)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = self.model(X_batch)
                all_preds.extend(outputs.squeeze().numpy())
                all_targets.extend(y_batch.numpy())

        mse = mean_squared_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        self.logger.info(f'Test MSE: {mse:.4f}, R2 Score: {r2:.4f}')
        return mse, r2
