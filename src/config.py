import os
from dotenv import load_dotenv
import hashlib

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Stores configuration variables and settings.
    """
    ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')
    OPENAI_API_KEY = os.getenv("open_ai")

    @staticmethod
    def create_config_id(target_variable, time_horizon, feature_subset, hyperparameters):
        """
        Create a concise config ID based on differentiating factors of each model.

        Args:
            target_variable (str): The target column being predicted.
            time_horizon (str): The prediction horizon (e.g., '5min', '1hour').
            feature_subset (list of str): List of feature column names included in the model.
            hyperparameters (dict): Dictionary of hyperparameters.

        Returns:
            str: A concise, unique identifier for the configuration.
        """
        # Create a string from differentiating factors
        feature_str = "_".join(sorted(feature_subset))
        hyperparam_str = "_".join([f"{k}={v}" for k, v in sorted(hyperparameters.items())])
        base_str = f"{target_variable}_{time_horizon}_{feature_str}_{hyperparam_str}"

        # Generate a hash of the string to keep it short and unique
        config_id = hashlib.md5(base_str.encode()).hexdigest()[:10]  # Use only the first 10 characters to make it short

        return config_id
