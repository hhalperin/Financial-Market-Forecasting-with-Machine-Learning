from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # API Keys and external service configurations.
    alphavantage_api_key: str
    openai_api_key: str

    # Storage configuration.
    storage_mode: str = "local"      # Options: "local", "s3", "db"
    local_mode: bool = True          # Set to False if using S3.
    s3_bucket: str = ""              # Required if local_mode is False.

    # Data and API parameters.
    ticker: str = "NVDA"
    start_date: str = "2022-01-01"
    end_date: str = "2025-01-01"
    interval: str = "1min"
    outputsize: str = "full"

    # File paths for data storage.
    data_storage_path: str = "./data"
    permanent_storage_path: str = "./permanent_storage"

    # DataEmbedder configuration.
    embedding_model_name: str = "gme-qwen2-vl2b"
    embedding_n_components: int = 128
    embedding_batch_size: int = 8
    embedding_use_pca: bool = True

    # SentimentProcessor configuration.
    sentiment_model_name: str = "ProsusAI/finbert"

    # TimeHorizonManager configuration.
    max_gather_minutes: int = 10080
    max_predict_minutes: int = 40320
    time_horizon_step: int = 1

    # Model training hyperparameters.
    model_learning_rate: float = 0.001
    model_weight_decay: float = 1e-4
    model_batch_size: int = 32
    model_epochs: int = 30
    model_hidden_layers: List[int] = [256, 128, 64]
    model_dropout_rate: float = 0.2
    model_loss_function: str = "smoothl1"  # Options: "mse", "smoothl1"

    # Learning rate scheduler configuration.
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2

    # Hyperparameter optimization configuration.
    hyperparameter_trials: int = 20000

    # Pipeline configuration.
    num_combos: int = 20000

    # Candidate architectures for model search.
    candidate_architectures: List[List[int]] = [[256, 128, 64], [128, 64, 32], [512, 256, 128], [1024, 512, 256, 128]]

    # Optional training filters.
    filter_sentiment: bool = True
    sentiment_threshold: float = 0.35
    filter_fluctuation: bool = False
    fluct_threshold: float = 1.0

    # Feature scaling configuration.
    use_scaler: bool = True
    scaler_type: str = "robust"  # Options: "robust", "standard", "minmax"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
