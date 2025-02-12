"""
Models Module

This module exposes the following classes and functions:
- StockPredictor: A neural network for stock prediction.
- ModelManager: Handles model creation, training, and evaluation.
- ModelAnalysis: Performs analysis on trained models including metric computation and feature importance.
- ModelPipeline: Coordinates training across multiple time horizon combinations.
- TrainingConfig: Data class for training experiment configuration.
- get_experiment_configurations: Returns a list of TrainingConfig objects for experiments.
"""

from .stock_predictor import StockPredictor
from .model_manager import ModelManager
from .model_analysis import ModelAnalysis
from .model_pipeline import ModelPipeline
from .configuration import TrainingConfig, get_experiment_configurations

__all__ = [
    "StockPredictor",
    "ModelManager",
    "ModelAnalysis",
    "ModelPipeline",
    "TrainingConfig",
    "get_experiment_configurations"
]
