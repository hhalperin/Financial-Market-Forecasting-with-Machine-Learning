import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from logger import get_logger
from utils.data_loader import get_data_loader

class ModelTrainer:
    """
    Trains the StockPredictor model.
    """

    def __init__(self, model, learning_rate=0.001, batch_size=32, epochs=50, saved_models_dir='saved_models'):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.logger = get_logger(self.__class__.__name__)
        self.saved_models_dir = saved_models_dir
        os.makedirs(self.saved_models_dir, exist_ok=True)

    def train(self, X_train, y_train, X_val, y_val, model_name='model', hyperparameters="default"):
        """
        Trains the model using the provided data and logs training metrics.
        """
        train_loader = get_data_loader(X_train, y_train, batch_size=self.batch_size, shuffle=True)
        val_loader = get_data_loader(X_val, y_val, batch_size=self.batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

        best_val_loss = float('inf')
        best_model_state = None
        early_stopping_counter = 0
        patience = 10  # Early stopping patience

        training_history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            training_history['train_loss'].append(avg_train_loss)
            self.logger.info(f'Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_train_loss:.4f}')

            # Validation phase
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = self.model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            training_history['val_loss'].append(avg_val_loss)
            self.logger.info(f'Epoch {epoch+1}/{self.epochs}, Validation Loss: {avg_val_loss:.4f}')

            # Adjust learning rate based on validation loss
            scheduler.step(avg_val_loss)

            # Save the best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict()
                early_stopping_counter = 0  # Reset early stopping counter
            else:
                early_stopping_counter += 1

            # Early stopping check
            if early_stopping_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1} with best validation loss {best_val_loss:.4f}")
                break

        # Save the best model
        if best_model_state is not None:
            best_model_path = os.path.join(self.saved_models_dir, model_name, "best_model.pth")
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(best_model_state, best_model_path)
            self.logger.info(f"Best model saved with validation loss {best_val_loss:.4f} to {best_model_path}")

        self.logger.info("Training completed.")
        return self.model, training_history
