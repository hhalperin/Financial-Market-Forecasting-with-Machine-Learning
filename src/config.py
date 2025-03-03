"""
Configuration Module

This module uses Pydantic Settings to centralize all configuration variables.
It loads configuration from environment variables (via a .env file) and provides
default values where applicable.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """
    Centralized configuration for the application. Settings are loaded from environment variables
    or use default values if not provided.
    """
    # API Keys and External Services
    alphavantage_api_key: str
    openai_api_key: str

    # Storage Configuration
    storage_mode: str = "local"      # Options: "local", "s3", "db"
    local_mode: bool = True          # Set to False if using S3.
    s3_bucket: str = ""              # Required if local_mode is False.

    # Data and API Parameters
    ticker: str = "HIMS"
    start_date: str = "2022-01-01"
    end_date: str = "2025-01-01"
    interval: str = "1min"
    outputsize: str = "full"

    # File Paths for Data Storage
    data_storage_path: str = "./src/data"
    permanent_storage_path: str = "./permanent_storage"

    # DataEmbedder Configuration
    embedding_model_name: str = "gme-qwen2-vl2b"
    embedding_n_components: int = 128
    embedding_batch_size: int = 8
    embedding_use_pca: bool = True
    embedding_combine_fields: bool = True
    embedding_fields_to_combine: List[str] = ["authors", "title", "summary"]
    embedding_combine_template: str = "authors: {authors}; title: {title}; summary: {summary}"

    # SentimentProcessor Configuration
    sentiment_model_name: str = "ProsusAI/finbert"
    sentiment_use_recency_weighting: bool = True
    sentiment_recency_decay: float = 0.01

    # TimeHorizonManager Configuration
    max_gather_minutes: int = 10080
    max_predict_minutes: int = 40320
    time_horizon_step: int = 5

    # Model Training Hyperparameters
    model_learning_rate: float = 0.001
    model_weight_decay: float = 1e-4
    model_batch_size: int = 32
    model_epochs: int = 30
    model_hidden_layers: List[int] = [256, 128, 64]
    model_dropout_rate: float = 0.2
    model_loss_function: str = "smoothl1"  # Options: "mse", "smoothl1"
    model_architecture: str = "feedforward"  # Options: "feedforward", "cnn"

    # Learning Rate Scheduler Configuration
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2

    # Hyperparameter Optimization Configuration
    hyperparameter_trials: int = 500

    # Pipeline Configuration
    num_combos: int = 500

    # Candidate Architectures for Model Search
    candidate_architectures: List[List[int]] = [
        [256, 128, 64],
        [128, 64, 32],
        [512, 256, 128],
        [1024, 512, 256, 128]
    ]

    # Optional Training Filters
    filter_sentiment: bool = True
    sentiment_threshold: float = 0.35
    filter_fluctuation: bool = False
    fluct_threshold: float = 1.0

    # Feature Scaling Configuration
    use_scaler: bool = True
    scaler_type: str = "robust"  # Options: "robust", "standard", "minmax"

    # Saving Behavior Configuration
    save_only_goated: bool = True  # If True, only save goated (best) model summaries and training data

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


# foobar build this

# from pydantic import BaseSettings, validator
# from typing import List

# class APISettings(BaseSettings):
#     alphavantage_api_key: str
#     openai_api_key: str

# class StorageSettings(BaseSettings):
#     storage_mode: str = "local"      # Options: "local", "s3", "db"
#     local_mode: bool = True          # Set to False if using S3.
#     s3_bucket: str = ""

#     @validator("s3_bucket")
#     def s3_bucket_required(cls, v, values):
#         if not values.get("local_mode") and not v:
#             raise ValueError("s3_bucket must be provided when local_mode is False")
#         return v

# class DataSettings(BaseSettings):
#     ticker: str = "HIMS"
#     start_date: str = "2022-01-01"
#     end_date: str = "2025-01-01"
#     interval: str = "1min"
#     outputsize: str = "full"
#     data_storage_path: str = "./src/data"
#     permanent_storage_path: str = "./permanent_storage"

# class EmbeddingSettings(BaseSettings):
#     embedding_model_name: str = "gme-qwen2-vl2b"
#     embedding_n_components: int = 128
#     embedding_batch_size: int = 8
#     embedding_use_pca: bool = True
#     embedding_combine_fields: bool = True
#     embedding_fields_to_combine: List[str] = ["authors", "title", "summary"]
#     embedding_combine_template: str = "authors: {authors}; title: {title}; summary: {summary}"

# class SentimentSettings(BaseSettings):
#     sentiment_model_name: str = "ProsusAI/finbert"
#     sentiment_use_recency_weighting: bool = True
#     sentiment_recency_decay: float = 0.01

# class TimeHorizonSettings(BaseSettings):
#     max_gather_minutes: int = 10080
#     max_predict_minutes: int = 40320
#     time_horizon_step: int = 5

# class TrainingSettings(BaseSettings):
#     model_learning_rate: float = 0.001
#     model_weight_decay: float = 1e-4
#     model_batch_size: int = 32
#     model_epochs: int = 30
#     model_hidden_layers: List[int] = [256, 128, 64]
#     model_dropout_rate: float = 0.2
#     model_loss_function: str = "smoothl1"  # Options: "mse", "smoothl1"
#     model_architecture: str = "feedforward"  # Options: "feedforward", "cnn"
#     lr_scheduler_factor: float = 0.5
#     lr_scheduler_patience: int = 2
#     hyperparameter_trials: int = 500

# class PipelineSettings(BaseSettings):
#     num_combos: int = 500
#     candidate_architectures: List[List[int]] = [
#         [256, 128, 64],
#         [128, 64, 32],
#         [512, 256, 128],
#         [1024, 512, 256, 128]
#     ]
#     filter_sentiment: bool = True
#     sentiment_threshold: float = 0.35
#     filter_fluctuation: bool = False
#     fluct_threshold: float = 1.0
#     use_scaler: bool = True
#     scaler_type: str = "robust"  # Options: "robust", "standard", "minmax"
#     save_only_goated: bool = True  # If True, only save goated (best) model summaries and training data

# class Settings(BaseSettings):
#     api: APISettings = APISettings()
#     storage: StorageSettings = StorageSettings()
#     data: DataSettings = DataSettings()
#     embedding: EmbeddingSettings = EmbeddingSettings()
#     sentiment: SentimentSettings = SentimentSettings()
#     time_horizon: TimeHorizonSettings = TimeHorizonSettings()
#     training: TrainingSettings = TrainingSettings()
#     pipeline: PipelineSettings = PipelineSettings()

#     class Config:
#         env_file = ".env"
#         env_file_encoding = "utf-8"

# settings = Settings()
