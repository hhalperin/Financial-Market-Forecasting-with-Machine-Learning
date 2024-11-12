import torch
import torch.nn as nn

class StockPredictor(nn.Module):
    """
    Defines the PyTorch model architecture for stock prediction.
    """

    def __init__(self, input_size, hidden_layers, output_size=1):
        super(StockPredictor, self).__init__()
        layers = []
        in_features = input_size
        for units in hidden_layers:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Optional dropout for regularization
            in_features = units
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
