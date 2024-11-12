import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.logger import get_logger

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

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the model using the provided data and logs training metrics.

        Returns:
            trained_model (nn.Module): The trained model.
            training_history (dict): Dictionary containing training and validation loss history.
        """
        # Prepare data loaders
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training history for visualization
        training_history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')  # Track the best validation loss

        # Training loop
        for epoch in range(self.epochs):
            # Training phase
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

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = os.path.join(self.saved_models_dir, f"best_model_epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"Best model saved at epoch {epoch+1} with validation loss {best_val_loss:.4f} to {model_path}")

        self.logger.info("Training completed.")
        return self.model, training_history
