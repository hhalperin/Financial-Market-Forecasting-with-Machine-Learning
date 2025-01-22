# src/data_aggregation/base_data_gatherer.py

import os
import json
import boto3
from src.utils.logger import get_logger
from src.utils.error_handler import handle_api_errors
import requests

class BaseDataGatherer:
    """
    Base class that fetches API keys and wraps requests with error handling.
    """

    def __init__(
        self,
        ticker,
        local_mode=False,
        api_key_env_var="ALPHAVANTAGE_API_KEY",
        secret_name_env_var="ALPHAVANTAGE_SECRET_NAME"
    ):
        """
        :param ticker: Stock ticker symbol, e.g. 'AAPL'.
        :param local_mode: If True, uses local environment for API key. Otherwise fetches from AWS Secrets Manager.
        :param api_key_env_var: Name of the environment variable that has the Alpha Vantage API key.
        :param secret_name_env_var: Name of the AWS Secret for the Alpha Vantage API key if not local.
        """
        self.ticker = ticker
        self.local_mode = local_mode
        self.logger = get_logger(self.__class__.__name__)
        self.api_key = self._get_api_key(api_key_env_var, secret_name_env_var)

    def _get_api_key(self, api_key_env_var, secret_name_env_var):
        """
        Retrieves the Alpha Vantage API key either from local environment var or AWS Secrets Manager.
        """
        if self.local_mode:
            self.logger.info("[LOCAL MODE] Using local environment variable for API key.")
            return os.getenv(api_key_env_var, "ALPHAVANTAGE_API_KEY")
        else:
            self.logger.info("[CLOUD MODE] Fetching API key from AWS Secrets Manager.")
            secret_name = os.environ[secret_name_env_var]
            session = boto3.session.Session()
            client = session.client(service_name='secretsmanager')
            secret_value = client.get_secret_value(SecretId=secret_name)
            creds = json.loads(secret_value['SecretString'])
            return creds[api_key_env_var]

    @handle_api_errors
    def make_api_request(self, url, headers=None):
        """
        Wraps requests.get with error handling & logging.
        :param url: Full request URL
        :param headers: Optional dict of headers
        :return: JSON payload as a dict
        """
        self.logger.info(f"Making API request: {url}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
