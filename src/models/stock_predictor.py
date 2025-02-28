import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class StockPredictor(nn.Module):
    """
    Feedforward neural network for stock price prediction.
    """
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int = 1, dropout_rate: float = 0.2) -> None:
        """
        Initializes the StockPredictor.
        
        :param input_size: Number of input features.
        :param hidden_layers: List of hidden layer sizes.
        :param output_size: Output dimension (default is 1).
        :param dropout_rate: Dropout rate for regularization.
        """
        super(StockPredictor, self).__init__()
        layers = []
        in_features = input_size
        for units in hidden_layers:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = units
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class StockCNNPredictor(nn.Module):
    """
    1D CNN model for stock price prediction with sequence input.
    """
    def __init__(self, input_size: int, seq_length: int, dropout_rate: float = 0.2):
        """
        Initializes the StockCNNPredictor.
        
        :param input_size: Number of input features.
        :param seq_length: Length of the input sequence.
        :param dropout_rate: Dropout rate.
        """
        super(StockCNNPredictor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (seq_length // 2), 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, seq_length, features]
        x = x.transpose(1, 2)  # Convert to [batch, features, seq_length] for Conv1d
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
