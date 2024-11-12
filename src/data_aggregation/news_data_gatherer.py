import requests
import pandas as pd
import logging
from datetime import datetime
from utils.config import Config
from utils.logger import get_logger
from utils.error_handler import handle_api_errors

class NewsDataGatherer:
    """
    Responsible for fetching news data from AlphaVantage API.
    """

    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = Config.ALPHAVANTAGE_API_KEY
        self.base_url = 'https://www.alphavantage.co/query'
        self.logger = get_logger(self.__class__.__name__)

    @handle_api_errors
    def fetch_news_data(self):
        """
        Fetches news articles for the given ticker and date range.
        """
        news_start_date = f"{self.start_date.replace('-', '')}T0000"
        news_end_date = f"{self.end_date.replace('-', '')}T2359"
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': self.ticker,
            'time_from': news_start_date,
            'time_to': news_end_date,
            'apikey': self.api_key,
            'sort': 'LATEST',
            'limit': 200
        }
        response = requests.get(self.base_url, params=params)
        data = response.json()

        if 'feed' in data:
            news_df = pd.DataFrame(data['feed'])
            news_df['Symbol'] = self.ticker
            news_df['time_published'] = pd.to_datetime(news_df['time_published'], format='%Y%m%dT%H%M%S')
            self.logger.info(f"Fetched news data for {self.ticker}")
            return news_df
        else:
            error_msg = data.get('Error Message', 'Unknown error occurred.')
            self.logger.error(f"Error fetching news data: {error_msg}")
            raise ValueError(error_msg)

    def run(self):
        """
        Orchestrates fetching of news data.
        """
        news_df = self.fetch_news_data()
        return news_df
