import torch
import torch.nn as nn

class StockPredictor(nn.Module):
    """
    A simple feedforward network for stock price change prediction.
    """

    def __init__(self, input_size, hidden_layers, output_size=1, dropout_rate=0.2):
        """
        :param input_size: Number of input features.
        :param hidden_layers: List of units in each hidden layer.
        :param output_size: Typically 1 for regression on price changes.
        :param dropout_rate: Dropout rate to help combat overfitting.
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

        # Final output layer
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass: x -> model -> output
        """
        return self.model(x)
