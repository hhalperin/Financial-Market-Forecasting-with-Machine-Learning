import requests
import pandas as pd
from config import Config
from logger import get_logger
from error_handler import handle_api_errors

class NewsDataGatherer:
    """
    Responsible for fetching news data from AlphaVantage API.
    """

    def __init__(self, ticker, start_date, end_date):
        """
        Initializes the NewsDataGatherer with necessary parameters.

        Args:
            ticker (str): The stock ticker symbol.
            start_date (str): Start date for fetching news data (in 'YYYY-MM-DD' format).
            end_date (str): End date for fetching news data (in 'YYYY-MM-DD' format).
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = Config.ALPHAVANTAGE_API_KEY  # API key for AlphaVantage.
        self.base_url = 'https://www.alphavantage.co/query'  # Base URL for AlphaVantage API.
        self.logger = get_logger(self.__class__.__name__)  # Get logger for the class.

    @handle_api_errors
    def fetch_news_data(self):
        """
        Fetches news articles for the given ticker and date range.

        Returns:
            pd.DataFrame: A DataFrame containing the news data.
        """
        # Construct the start and end date strings in the format expected by the API.
        news_start_date = f"{self.start_date.replace('-', '')}T0000"
        news_end_date = f"{self.end_date.replace('-', '')}T000"

        # Set up the parameters for the API request.
        params = {
            'function': 'NEWS_SENTIMENT',  # API function for fetching news sentiment.
            'tickers': self.ticker,  # The stock ticker symbol.
            'time_from': news_start_date,  # Start time for news articles.
            'time_to': news_end_date,  # End time for news articles.
            'apikey': self.api_key,  # API key for authentication.
            'sort': 'LATEST',  # Sort news articles by the latest.
            'limit': 1000  # Limit the number of articles fetched.
        }

        # Make the request to the AlphaVantage API.
        response = requests.get(self.base_url, params=params)

        # Log the full request URL for debugging purposes.
        full_url = response.url

        # Convert the response to a JSON object.
        data = response.json()

        # If the response contains news data, process it.
        if 'feed' in data:
            # Convert the news feed to a DataFrame.
            news_df = pd.DataFrame(data['feed'])
            news_df['Symbol'] = self.ticker  # Add the stock ticker symbol to the DataFrame.
            # Convert the publication time to datetime format.
            news_df['time_published'] = pd.to_datetime(news_df['time_published'], format='%Y%m%dT%H%M%S')
            self.logger.info(f"Fetched news data for {self.ticker}")  # Log successful fetch.
            return news_df
        else:
            # Log the error if the response does not contain the expected news data.
            error_msg = data.get('Error Message', 'Unknown error occurred.')
            self.logger.error(f"Error fetching news data: {error_msg}. Full URL: {full_url}")
            # Raise an error to indicate that news data could not be fetched.
            raise ValueError(f"Error fetching data: {error_msg}", full_url)

    def run(self):
        """
        Function that can be called to fetch news data.

        Returns:
            pd.DataFrame: A DataFrame containing the news data.
        """
        return self.fetch_news_data()
