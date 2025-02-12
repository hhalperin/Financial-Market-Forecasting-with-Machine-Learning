"""
News Data Gatherer Module

Gathers news articles from Alpha Vantage over a specified date range by splitting the period
into annual chunks to manage API limitations.
"""

import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Tuple
from src.data_aggregation.base_data_gatherer import BaseDataGatherer
from src.utils.logger import get_logger


class NewsDataGatherer(BaseDataGatherer):
    """
    Gathers news articles for a given ticker over a specified date range.
    Splits the date range into manageable annual chunks.
    """

    def __init__(self, ticker: str, start_date: str, end_date: str, local_mode: bool = False) -> None:
        """
        Initializes the NewsDataGatherer.

        :param ticker: Stock ticker symbol (e.g., 'TSLA').
        :param start_date: Start date in 'YYYY-MM-DD' format.
        :param end_date: End date in 'YYYY-MM-DD' format.
        :param local_mode: Flag to indicate if the application runs in local mode.
        """
        super().__init__(ticker, local_mode=local_mode)
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.base_url: str = 'https://www.alphavantage.co/query'
        self.logger = get_logger(self.__class__.__name__)

    def _generate_yearly_ranges(self) -> List[Tuple[datetime, datetime]]:
        """
        Generates a list of yearly date ranges (as tuples) within the specified date range.
        For a date range shorter than a year, returns a single range.

        :return: List of (start_datetime, end_datetime) tuples.
        """
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        ranges: List[Tuple[datetime, datetime]] = []
        current_start = start_dt

        while current_start <= end_dt:
            # Define the end of the current chunk as December 31 or the actual end date.
            chunk_end = datetime(current_start.year, 12, 31)
            if chunk_end > end_dt:
                chunk_end = end_dt
            ranges.append((current_start, chunk_end))
            # Move to January 1 of the next year.
            current_start = datetime(current_start.year + 1, 1, 1)

        return ranges

    def _fetch_news_for_range(self, range_start: datetime, range_end: datetime) -> pd.DataFrame:
        """
        Fetches up to 1000 news articles for the specified datetime range using the Alpha Vantage API.
        Formats the time parameters as required and logs the process.

        :param range_start: Start datetime of the range.
        :param range_end: End datetime of the range.
        :return: DataFrame containing the news articles for the given range.
        """
        news_start_str = range_start.strftime("%Y%m%dT0000")
        news_end_str = range_end.strftime("%Y%m%dT2359")
        url = (
            f"{self.base_url}?function=NEWS_SENTIMENT"
            f"&tickers={self.ticker}"
            f"&limit=1000"
            f"&sort=LATEST"
            f"&time_from={news_start_str}"
            f"&time_to={news_end_str}"
            f"&apikey={self.api_key}"
        )
        # self.logger.debug(f"Fetching news for range {news_start_str} to {news_end_str} using URL: {url}")
        data = self.make_api_request(url)

        if 'feed' not in data:
            self.logger.error(f"Error fetching news for range {range_start} to {range_end}: {data}")
            return pd.DataFrame()

        df = pd.DataFrame(data['feed'])
        df['Symbol'] = self.ticker
        # Convert the published time string into a datetime object.
        df['time_published'] = pd.to_datetime(df['time_published'], format='%Y%m%dT%H%M%S', errors='coerce')
        self.logger.info(f"Fetched {len(df)} news records for {self.ticker} from {news_start_str} to {news_end_str}")
        return df

    def _fetch_news_data_in_chunks(self) -> pd.DataFrame:
        """
        Iterates over each yearly range, fetches the corresponding news data,
        and concatenates the results into a single DataFrame.

        :return: A concatenated DataFrame containing news data for all chunks.
        """
        year_ranges = self._generate_yearly_ranges()
        all_dfs: List[pd.DataFrame] = []

        for (start_dt, end_dt) in year_ranges:
            df_chunk = self._fetch_news_for_range(start_dt, end_dt)
            if not df_chunk.empty:
                all_dfs.append(df_chunk)

        if not all_dfs:
            self.logger.warning("No news articles retrieved in any chunk.")
            return pd.DataFrame()

        big_df = pd.concat(all_dfs, ignore_index=True)
        big_df.sort_values("time_published", inplace=True)
        big_df.drop_duplicates(subset=["time_published", "title"], inplace=True)
        big_df.reset_index(drop=True, inplace=True)
        return big_df

    def run(self) -> pd.DataFrame:
        """
        Main entry point to retrieve news data over the specified date range.
        Drops rows missing essential columns ('title' or 'summary') if available.

        :return: DataFrame containing the processed news data.
        """
        df = self._fetch_news_data_in_chunks()
        if df.empty:
            return df

        # Check if essential columns are present before dropping null values.
        missing_cols = [col for col in ["title", "summary"] if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing columns {missing_cols}. Skipping dropna for these columns.")
        else:
            initial_count = len(df)
            df.dropna(subset=["title", "summary"], inplace=True)
            self.logger.info(f"Dropped {initial_count - len(df)} rows missing 'title' or 'summary'.")

        return df
