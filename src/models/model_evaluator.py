import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from logger import get_logger
from utils.data_loader import get_data_loader

class ModelEvaluator:
    """
    Evaluates the trained model on test data.
    """

    def __init__(self, model):
        self.model = model
        self.logger = get_logger(self.__class__.__name__)

    def evaluate(self, X_test, y_test, batch_size=32):
        """
        Evaluates model performance.
        """
        self.model.eval()

        test_loader = get_data_loader(X_test, y_test, batch_size=batch_size, shuffle=False)
        all_preds, all_targets = [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = self.model(X_batch)
                all_preds.extend(outputs.squeeze().cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)

        self.logger.info(f'Test MSE: {mse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}')
        return mse, mae, r2
