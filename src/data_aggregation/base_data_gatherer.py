import os
import json
import boto3
from utils.logger import get_logger
from utils.error_handler import handle_api_errors
import requests

class BaseDataGatherer:
    def __init__(self, ticker, local_mode=False, api_key_env_var="ALPHAVANTAGE_API_KEY", secret_name_env_var="ALPHAVANTAGE_SECRET_NAME"):
        self.ticker = ticker
        self.local_mode = local_mode
        self.logger = get_logger(self.__class__.__name__)
        self.api_key = self._get_api_key(api_key_env_var, secret_name_env_var)

    def _get_api_key(self, api_key_env_var, secret_name_env_var):
        if self.local_mode:
            self.logger.info(f"[LOCAL MODE] Using local environment variable for API key.")
            return os.getenv(api_key_env_var, "LOCAL_TEST_KEY")
        else:
            self.logger.info(f"[CLOUD MODE] Fetching API key from AWS Secrets Manager.")
            secret_name = os.environ[secret_name_env_var]
            session = boto3.session.Session()
            client = session.client(service_name='secretsmanager')
            secret_value = client.get_secret_value(SecretId=secret_name)
            creds = json.loads(secret_value['SecretString'])
            return creds[api_key_env_var]

    @handle_api_errors
    def make_api_request(self, url, params):
        self.logger.info(f"Making API request to {url} with params: {params}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()