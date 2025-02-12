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
    data_storage_path: str = "./data"          # Directory for temporary storage.
    permanent_storage_path: str = "./permanent_storage"  # Directory for permanent storage.

    # DataEmbedder configuration.
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_n_components: int = 128
    embedding_batch_size: int = 8
    embedding_use_pca: bool = True

    # SentimentProcessor configuration.
    sentiment_model_name: str = "ProsusAI/finbert"

    # TimeHorizonManager configuration.
    max_gather_minutes: int = 2880     # Maximum gather time in minutes.
    max_predict_minutes: int = 10080   # Maximum prediction time in minutes.
    time_horizon_step: int = 1         # Step size in minutes.

    # Model training hyperparameters.
    model_learning_rate: float = 0.001
    model_batch_size: int = 32
    model_epochs: int = 30
    model_hidden_layers: List[int] = [256, 128, 64]

    # Optional training filters (can be used by the model pipeline).
    filter_sentiment: bool = True
    sentiment_threshold: float = 0.35
    filter_fluctuation: bool = False
    fluct_threshold: float = 1.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# For convenience, you can instantiate a global Settings object:
settings = Settings()
