# src/data_aggregation/news_data_gatherer.py

import pandas as pd
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from src.data_aggregation.base_data_gatherer import BaseDataGatherer

class NewsDataGatherer(BaseDataGatherer):
    """
    Gathers news articles from Alpha Vantage for a given date range, 
    splitting into annual chunks to retrieve more data.
    """

    def __init__(self, ticker, start_date, end_date, local_mode=False):
        """
        :param ticker: e.g. 'AAPL'
        :param start_date: 'YYYY-MM-DD'
        :param end_date:   'YYYY-MM-DD'
        :param local_mode: If True, local env for API key; otherwise fetch from Secrets Manager
        """
        super().__init__(ticker, local_mode=local_mode)
        self.start_date = start_date
        self.end_date = end_date
        self.base_url = 'https://www.alphavantage.co/query'

    def _generate_yearly_ranges(self):
        """
        Yields a list of (year_start_str, year_end_str) in 'YYYY-MM-DD' format.
        For example, if start_date=2022-01-10 and end_date=2024-03-05, 
        we might get:
          - [2022-01-10, 2022-12-31]
          - [2023-01-01, 2023-12-31]
          - [2024-01-01, 2024-03-05]
        We clamp to each Dec 31 or final end date.
        """
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")

        ranges = []
        current_start = start_dt

        while current_start <= end_dt:
            # End of the current chunk = Dec 31 of current_start's year
            chunk_end = datetime(current_start.year, 12, 31)
            if chunk_end > end_dt:
                # clamp to actual end_dt
                chunk_end = end_dt

            ranges.append((current_start, chunk_end))

            # Next chunk starts Jan 1 of the following year
            next_year_start = datetime(current_start.year + 1, 1, 1)
            current_start = next_year_start

        return ranges

    def _fetch_news_for_range(self, range_start, range_end):
        """
        Fetch up to 1000 news articles for [range_start, range_end].
        :param range_start, range_end: datetime objects
        :return: DataFrame of news
        """
        # Convert to 'YYYYMMDDTHHMM'
        news_start_str = range_start.strftime("%Y%m%dT0000")
        news_end_str   = range_end.strftime("%Y%m%dT2359")  # e.g. 20221231T2359

        url = (
            f"{self.base_url}?function=NEWS_SENTIMENT"
            f"&tickers={self.ticker}"
            f"&limit=1000"
            f"&time_from={news_start_str}"
            f"&time_to={news_end_str}"
            f"&apikey={self.api_key}"
        )

        data = self.make_api_request(url)
        if 'feed' not in data:
            error_msg = data.get('Error Message', 'No news feed returned')
            self.logger.error(f"Error fetching news for range {range_start} to {range_end}: {error_msg}")
            return pd.DataFrame()

        df = pd.DataFrame(data['feed'])
        df['Symbol'] = self.ticker
        # Convert 'time_published'
        df['time_published'] = pd.to_datetime(df['time_published'], format='%Y%m%dT%H%M%S', errors='coerce')
        self.logger.info(f"Fetched {len(df)} news records for {self.ticker} from {news_start_str} to {news_end_str}")
        return df

    def _fetch_news_data_in_chunks(self):
        """
        Loops over each yearly range, fetches up to 1000 articles per chunk, 
        and concatenates them. 
        """
        year_ranges = self._generate_yearly_ranges()
        all_dfs = []
        for (start_dt, end_dt) in year_ranges:
            df_chunk = self._fetch_news_for_range(start_dt, end_dt)
            if not df_chunk.empty:
                all_dfs.append(df_chunk)

        if not all_dfs:
            self.logger.warning("No news articles retrieved in any chunk.")
            return pd.DataFrame()

        big_df = pd.concat(all_dfs, ignore_index=True)
        # Sort by time_published, remove duplicates
        big_df.sort_values("time_published", inplace=True)
        big_df.drop_duplicates(subset=["time_published", "title"], inplace=True)
        big_df.reset_index(drop=True, inplace=True)
        return big_df

    def run(self):
        """
        Main method to retrieve news data across yearly chunks, 
        then filter out rows missing 'title' or 'summary'.
        """
        df = self._fetch_news_data_in_chunks()
        if df.empty:
            return df

        # Remove rows missing 'title' or 'summary' (assuming these columns exist)
        # If 'summary' doesn't always exist, you can do a safe check or handle KeyError.
        missing_cols = [col for col in ["title", "summary"] if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing columns {missing_cols}. Can't drop them.")
        else:
            # Drop rows that have NaN in either 'title' or 'summary'
            initial_count = len(df)
            df.dropna(subset=["title", "summary"], inplace=True)
            self.logger.info(f"Dropped {initial_count - len(df)} rows missing title or summary.")

        return df
