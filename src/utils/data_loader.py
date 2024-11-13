# utils/data_loader_utils.py

import torch
from torch.utils.data import DataLoader, TensorDataset

def get_data_loader(X, y, batch_size=32, shuffle=True):
    """
    Create a DataLoader from input features and labels.

    Args:
        X (np.ndarray or torch.Tensor): Input features.
        y (np.ndarray or torch.Tensor): Target labels.
        batch_size (int): Batch size for data loading.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: A PyTorch DataLoader instance for the input data.
    """
    # Convert inputs to torch tensors if needed
    X_tensor = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
    y_tensor = torch.tensor(y, dtype=torch.float32) if not isinstance(y, torch.Tensor) else y

    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader
