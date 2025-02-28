"""
Data Loader Module

Provides a helper function to create a PyTorch DataLoader from input features and target labels.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Union
import numpy as np

def get_data_loader(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """
    Creates a DataLoader from input features and target labels.
    
    :param X: Input features as a NumPy array or torch.Tensor.
    :param y: Target labels as a NumPy array or torch.Tensor.
    :param batch_size: Batch size for the DataLoader.
    :param shuffle: Whether to shuffle the data.
    :return: A PyTorch DataLoader.
    """
    if not isinstance(X, torch.Tensor):
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        X_tensor = X
    if not isinstance(y, torch.Tensor):
        y_tensor = torch.tensor(y, dtype=torch.float32)
    else:
        y_tensor = y
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
