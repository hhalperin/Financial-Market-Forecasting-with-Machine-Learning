import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils.logger import get_logger
from utils.data_loader import get_data_loader

class ModelEvaluator:
    def __init__(self, model, data_handler=None, model_stage='models'):
        """
        :param model: Trained PyTorch model for evaluation
        :param data_handler: Optional DataHandler instance for storing metrics
        :param model_stage: The storage stage/folder for saving metrics
        """
        self.model = model
        self.data_handler = data_handler
        self.model_stage = model_stage
        self.logger = get_logger(self.__class__.__name__)

    def evaluate(self, X_test, y_test, batch_size=32, model_name='model'):
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
        self.logger.info(f'Test MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')

        # Store only serializable metrics, excluding training_history
        if self.data_handler is not None:
            metrics = {"mse": float(mse), "mae": float(mae), "r2": float(r2)}
            metrics_filename = f"{model_name}_metrics.json"
            self.data_handler.save_metrics_json(metrics, metrics_filename, stage=self.model_stage)

        return mse, mae, r2
