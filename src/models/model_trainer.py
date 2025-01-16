import io
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.logger import get_logger
from utils.data_loader import get_data_loader
from tqdm import tqdm

class ModelTrainer:
    """
    Handles the training loop for a PyTorch model, saving the best model
    to either local or S3 (depending on DataHandler config).
    """

    def __init__(self, model, learning_rate=0.001, batch_size=32, epochs=50,
                 data_handler=None, model_stage="models"):
        """
        :param model: PyTorch model (e.g., StockPredictor)
        :param data_handler: For saving best model weights in local or S3
        :param model_stage: S3/local subfolder to store model artifacts (default='models')
        """
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_handler = data_handler
        self.model_stage = model_stage  # 'models' subfolder
        self.logger = get_logger(self.__class__.__name__)

    def train(self, X_train, y_train, X_val, y_val, model_name='model'):
        import torch  # Explicit local import to fix the UnboundLocalError

        # 1) Build DataLoaders
        train_loader = get_data_loader(X_train, y_train, batch_size=self.batch_size, shuffle=True)
        val_loader = get_data_loader(X_val, y_val, batch_size=self.batch_size, shuffle=False)

        # 2) Define loss/optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        learning_rate = scheduler.get_last_lr()
        best_val_loss = float('inf')
        best_model_state = None
        early_stopping_counter = 0
        patience = 10

        training_history = {'train_loss': [], 'val_loss': []}

        # 3) Training loop
        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0.0

            epoch_desc = f"Epoch {epoch+1}/{self.epochs} [Train]"
            for X_batch, y_batch in tqdm(train_loader, desc=epoch_desc, leave=False):
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            training_history['train_loss'].append(avg_train_loss)
            self.logger.info(f'[Epoch {epoch+1}/{self.epochs}] Train Loss: {avg_train_loss:.4f}')

            # Validation
            self.model.eval()
            total_val_loss = 0.0
            for X_batch, y_batch in val_loader:
                with torch.no_grad():
                    outputs = self.model(X_batch)
                    val_loss = criterion(outputs.squeeze(), y_batch)
                    total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            training_history['val_loss'].append(avg_val_loss)
            self.logger.info(f'[Epoch {epoch+1}/{self.epochs}] Val Loss: {avg_val_loss:.4f}')

            scheduler.step(avg_val_loss)

            # Check for best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # 4) Save best model to local or S3 using data_handler
        if best_model_state is not None and self.data_handler is not None:
            buffer = io.BytesIO()
            torch.save(best_model_state, buffer)
            buffer.seek(0)
            self.data_handler.save_model_bytes(buffer.read(), f"{model_name}_best_model.pth", stage=self.model_stage)
            self.logger.info(f"Best model saved in stage='{self.model_stage}' as {model_name}_best_model.pth")

        self.logger.info("Training completed.")
        return self.model, training_history
