import pandas as pd
from .base_data_gatherer import BaseDataGatherer

class NewsDataGatherer(BaseDataGatherer):
    def __init__(self, ticker, start_date, end_date, local_mode=False):
        super().__init__(ticker, local_mode)
        self.start_date = start_date
        self.end_date = end_date
        self.base_url = 'https://www.alphavantage.co/query'

    def fetch_news_data(self):
        news_start_date = f"{self.start_date.replace('-','')}T0000"
        news_end_date = f"{self.end_date.replace('-','')}T0000"
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': self.ticker,
            'time_from': news_start_date,
            'time_to': news_end_date,
            'apikey': self.api_key,
            'sort': 'LATEST',
            'limit': 1000
        }
        data = self.make_api_request(self.base_url, params)
        if 'feed' in data:
            df = pd.DataFrame(data['feed'])
            df['Symbol'] = self.ticker
            df['time_published'] = pd.to_datetime(df['time_published'], format='%Y%m%dT%H%M%S')
            self.logger.info(f"Fetched {len(df)} news records for {self.ticker}.")
            return df
        else:
            error_msg = data.get('Error Message', 'Unknown error')
            self.logger.error(f"Error fetching news: {error_msg}")
            return pd.DataFrame()

    def run(self):
        return self.fetch_news_data()
