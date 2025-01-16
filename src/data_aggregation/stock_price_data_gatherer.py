import pandas as pd
from .base_data_gatherer import BaseDataGatherer

class StockPriceDataGatherer(BaseDataGatherer):
    def __init__(self, ticker, interval='5min', outputsize='full', month=None, local_mode=False):
        super().__init__(ticker, local_mode)
        self.interval = interval
        self.outputsize = outputsize
        self.month = month
        self.base_url = 'https://www.alphavantage.co/query'

    def fetch_intraday_data(self):
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': self.ticker,
            'interval': self.interval,
            'outputsize': self.outputsize,
            'apikey': self.api_key
        }
        data = self.make_api_request(self.base_url, params)
        key = f"Time Series ({self.interval})"
        if key in data:
            df = pd.DataFrame.from_dict(data[key], orient='index')
            return self._process_price_dataframe(df)
        else:
            error_msg = data.get('Note') or data.get('Error Message') or 'Unknown error'
            self.logger.error(f"Error fetching price data: {error_msg}")
            return pd.DataFrame()

    def _process_price_dataframe(self, df):
        df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }, inplace=True)
        df['Symbol'] = self.ticker
        df.index = pd.to_datetime(df.index)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'DateTime'}, inplace=True)
        return df[['Symbol', 'DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']]

    def run(self):
        return self.fetch_intraday_data()
