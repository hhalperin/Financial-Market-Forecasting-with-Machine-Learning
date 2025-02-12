"""
Base Data Gatherer Module

Provides the BaseDataGatherer class and helper functions to manage API key retrieval
and to make HTTP requests with retry logic.
"""

import os
import json
import boto3
import requests
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_fixed
from src.utils.logger import get_logger

# Global cache for API keys to avoid redundant lookups.
_CACHED_API_KEYS: Dict[tuple, str] = {}


def get_cached_api_key(api_key_env_var: str, secret_name_env_var: str, local_mode: bool) -> str:
    """
    Retrieves and caches the API key from either environment variables (local mode)
    or AWS Secrets Manager (production mode).

    :param api_key_env_var: The environment variable name for the API key.
    :param secret_name_env_var: The environment variable name for the secret's name in AWS Secrets Manager.
    :param local_mode: Boolean indicating whether the application is running in local mode.
    :return: The retrieved API key as a string.
    """
    key = (api_key_env_var, secret_name_env_var, local_mode)
    if key in _CACHED_API_KEYS:
        return _CACHED_API_KEYS[key]

    if local_mode:
        # In local mode, retrieve the API key directly from environment variables.
        value = os.getenv(api_key_env_var, "ALPHAVANTAGE_API_KEY")
    else:
        # In production, fetch the API key from AWS Secrets Manager.
        secret_name = os.environ[secret_name_env_var]
        session = boto3.session.Session()
        client = session.client(service_name='secretsmanager')
        secret_value = client.get_secret_value(SecretId=secret_name)
        creds = json.loads(secret_value['SecretString'])
        value = creds[api_key_env_var]

    _CACHED_API_KEYS[key] = value
    return value


class BaseDataGatherer:
    """
    Base class for data gatherers. Provides common functionality such as API key retrieval
    and making API requests with retry logic.
    """

    def __init__(self, ticker: str, local_mode: bool = False,
                 api_key_env_var: str = "ALPHAVANTAGE_API_KEY",
                 secret_name_env_var: str = "ALPHAVANTAGE_SECRET_NAME") -> None:
        """
        Initializes the data gatherer with the required API credentials and a persistent HTTP session.

        :param ticker: Stock ticker symbol.
        :param local_mode: Flag to indicate local mode (for API key retrieval).
        :param api_key_env_var: Environment variable name for the API key.
        :param secret_name_env_var: Environment variable name for the AWS secret name.
        """
        self.ticker: str = ticker
        self.local_mode: bool = local_mode
        self.logger = get_logger(self.__class__.__name__)
        self.api_key: str = get_cached_api_key(api_key_env_var, secret_name_env_var, local_mode)
        # Create a persistent HTTP session for better performance.
        self.session: requests.Session = requests.Session()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def make_api_request(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Makes an HTTP GET request using a persistent session. Retries the request upon failure.

        :param url: The full URL for the API request.
        :param headers: Optional HTTP headers for the request.
        :return: The JSON response as a dictionary.
        """
        # self.logger.info(f"Making API request: {url}")
        response = self.session.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
