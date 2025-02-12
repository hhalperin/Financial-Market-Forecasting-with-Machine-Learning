"""
Stock Predictor Module

Defines a simple feedforward neural network for stock price change prediction.
"""

import torch
import torch.nn as nn
from typing import List

class StockPredictor(nn.Module):
    """
    A simple feedforward neural network for regression tasks.
    """
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int = 1, dropout_rate: float = 0.2) -> None:
        """
        Initializes the StockPredictor.

        :param input_size: Number of input features.
        :param hidden_layers: List specifying the number of units in each hidden layer.
        :param output_size: Number of output units (default is 1 for regression).
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
        """
        Forward pass through the network.
        """
        return self.model(x)
