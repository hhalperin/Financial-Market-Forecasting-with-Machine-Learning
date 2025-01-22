# src/data_aggregation/stock_price_data_gatherer.py

import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from src.data_aggregation.base_data_gatherer import BaseDataGatherer

class StockPriceDataGatherer(BaseDataGatherer):
    """
    Gathers intraday stock price data from Alpha Vantage by splitting
    the date range into monthly chunks, appending 'month=YYYY-MM'
    in the URL query, just like your old StockDataPull_AlphaVantage code.
    """

    def __init__(
        self,
        ticker,
        start_date,
        end_date,
        interval='1min',
        local_mode=False
    ):
        """
        :param ticker: e.g. 'AAPL'
        :param start_date: 'YYYY-MM-DD'
        :param end_date: 'YYYY-MM-DD'
        :param interval: '1min', '5min', etc.
        :param local_mode: If True, read API key from local env var; else fetch from AWS Secrets Manager.
        """
        super().__init__(ticker, local_mode=local_mode)
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.base_url = "https://www.alphavantage.co/query"

    def _generate_month_params(self):
        """
        Returns a list of 'month=YYYY-MM' strings for each month from start_date to end_date.
        """
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')
        date_list = []
        current_date = start_dt
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y-%m')
            date_list.append(f"month={date_str}")
            current_date = current_date + relativedelta(months=1)
        return date_list

    def _fetch_monthly_data(self):
        """
        Loops over each monthly param, builds the request URL,
        fetches JSON, converts to DataFrame, and concatenates them.
        """
        date_list = self._generate_month_params()
        df_list = []

        for date_frag in date_list:
            # Example URL: 
            # https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY
            #   &symbol=NVDA &interval=1min &month=2022-05 &outputsize=full &apikey=XYZ
            url = (
                f"{self.base_url}?function=TIME_SERIES_INTRADAY"
                f"&symbol={self.ticker}"
                f"&interval={self.interval}"
                f"&{date_frag}"
                f"&outputsize=full"
                f"&apikey={self.api_key}"
            )

            data = self.make_api_request(url)
            key = f"Time Series ({self.interval})"
            ts_data = data.get(key, {})
            if not ts_data:
                self.logger.warning(f"No intraday data returned for {date_frag}")
                continue

            df = pd.DataFrame.from_dict(ts_data, orient='index')
            df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            }, inplace=True)
            df['Symbol'] = self.ticker
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'DateTime'}, inplace=True)
            df_list.append(df)

        if not df_list:
            return pd.DataFrame()

        # Concatenate
        pricing_df = pd.concat(df_list, ignore_index=True)
        pricing_df = pricing_df[['Symbol', 'DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        # Sort, deduplicate
        pricing_df['DateTime'] = pd.to_datetime(pricing_df['DateTime'], errors='coerce')
        pricing_df.sort_values('DateTime', inplace=True)
        pricing_df.drop_duplicates(subset=['DateTime'], inplace=True)
        pricing_df.reset_index(drop=True, inplace=True)
        return pricing_df

    def run(self):
        """
        Main entry point: fetch intraday data in monthly chunks.
        """
        return self._fetch_monthly_data()
