Help me organize my code and comment it to so that it is easy to explain to other people when I go through it please. I will give you pieces of the codebase, organized into directories, so that you can carefully review it and provide suggestions. Before you give me suggestions, review each file and make sure you provide a comprehensive plan for organizing the code. Then provide each refactored file entirely. 

Here is the first directory, data_aggregation:

First file:
__init__.py:
"""
Data Aggregation Module

This module provides classes to aggregate stock price data and news data.
"""

from .data_aggregator import DataAggregator
from .news_data_gatherer import NewsDataGatherer
from .stock_price_data_gatherer import StockPriceDataGatherer
from .base_data_gatherer import BaseDataGatherer

__all__ = [
    "DataAggregator",
    "NewsDataGatherer",
    "BaseDataGatherer",
    "StockPriceDataGatherer"
]

Second file:
base_data_gatherer.py:
import os
import json
import boto3
from typing import Optional, Dict, Any
import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from src.utils.logger import get_logger

# Global cache for API keys
_CACHED_API_KEYS: Dict[tuple, str] = {}

def get_cached_api_key(api_key_env_var: str, secret_name_env_var: str, local_mode: bool) -> str:
    """
    Retrieves and caches the API key either from the environment or from AWS Secrets Manager.
    """
    key = (api_key_env_var, secret_name_env_var, local_mode)
    if key in _CACHED_API_KEYS:
        return _CACHED_API_KEYS[key]
    if local_mode:
        value = os.getenv(api_key_env_var, "ALPHAVANTAGE_API_KEY")
    else:
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
    Base class for fetching API keys and making API requests with error handling.
    """
    def __init__(self, ticker: str, local_mode: bool = False,
                 api_key_env_var: str = "ALPHAVANTAGE_API_KEY",
                 secret_name_env_var: str = "ALPHAVANTAGE_SECRET_NAME") -> None:
        self.ticker: str = ticker
        self.local_mode: bool = local_mode
        self.logger = get_logger(self.__class__.__name__)
        self.api_key: str = get_cached_api_key(api_key_env_var, secret_name_env_var, local_mode)
        # Use a persistent requests session to improve performance.
        self.session: requests.Session = requests.Session()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def make_api_request(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Makes an API request using a persistent session with retry logic.
        
        :param url: Full request URL.
        :param headers: Optional HTTP headers.
        :return: JSON response as a dictionary.
        """
        self.logger.info(f"Making API request: {url}")
        response = self.session.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

Third file:
data_aggregator.py:
 import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
from src.utils.logger import get_logger
from src.data_aggregation.stock_price_data_gatherer import StockPriceDataGatherer
from src.data_aggregation.news_data_gatherer import NewsDataGatherer

logger = get_logger("DataAggregator")

class DataAggregator:
    """
    Aggregates intraday stock price data and news articles for a given ticker and date range.
    """
    def __init__(self, ticker: str, start_date: str, end_date: str,
                 interval: str = '1min', data_handler=None, local_mode: bool = False) -> None:
        self.ticker: str = ticker
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.interval: str = interval
        self.data_handler = data_handler
        self.local_mode: bool = local_mode

    def _fetch_price_data(self) -> pd.DataFrame:
        """
        Fetches intraday price data using the StockPriceDataGatherer.
        """
        gatherer = StockPriceDataGatherer(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            interval=self.interval,
            local_mode=self.local_mode
        )
        return gatherer.run()

    def _fetch_news_data(self) -> pd.DataFrame:
        """
        Fetches news data using the NewsDataGatherer.
        """
        gatherer = NewsDataGatherer(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            local_mode=self.local_mode
        )
        return gatherer.run()

    def aggregate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetches price and news data concurrently and returns them.
        """
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_price = executor.submit(self._fetch_price_data)
            future_news = executor.submit(self._fetch_news_data)
            price_df = future_price.result()
            news_df = future_news.result()
        elapsed = time.time() - start_time
        logger.info(f"Data aggregation completed in {elapsed:.2f}s.")
        return price_df, news_df

Forth file:
main.py:
import os
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.utils.data_handler import DataHandler
from src.data_aggregation.data_aggregator import DataAggregator
from src.config import Settings  # Centralized configuration
import pandas as pd

logger = get_logger("DataAggregationMain")

def load_config() -> dict:
    """
    Loads configuration from the Settings model.
    """
    settings = Settings()
    config = {
        "local_mode": settings.local_mode,
        "storage_mode": "local" if settings.local_mode else "s3",
        "s3_bucket": settings.s3_bucket if not settings.local_mode else None,
        "ticker": settings.ticker,
        "start_date": settings.start_date,
        "end_date": settings.end_date,
        "interval": settings.interval
    }
    return config

def setup_data_handler(config: dict) -> DataHandler:
    if config["local_mode"]:
        logger.info("[SETUP] Using local file storage.")
        return DataHandler(storage_mode="local")
    else:
        logger.info(f"[SETUP] Using S3 file storage with bucket: {config['s3_bucket']}")
        return DataHandler(bucket=config["s3_bucket"], storage_mode="s3")

def setup_aggregator(config: dict, data_handler: DataHandler) -> DataAggregator:
    return DataAggregator(
        ticker=config["ticker"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        interval=config["interval"],
        data_handler=data_handler,
        local_mode=config["local_mode"]
    )

def main() -> None:
    logger.info("[MAIN] Loading configuration...")
    config = load_config()
    logger.info(f"[MAIN] Configuration loaded: {config}")

    data_handler = setup_data_handler(config)
    aggregator = setup_aggregator(config, data_handler)
    logger.info("[MAIN] Starting data aggregation (monthly intraday + news)...")
    price_df, news_df = aggregator.aggregate_data()
    logger.info("[MAIN] Data aggregation process completed.")

    ticker = config["ticker"]
    start_date = config["start_date"]
    end_date = config["end_date"]

    if price_df is not None and not price_df.empty:
        price_filename = f"{ticker}_price_{start_date}_to_{end_date}.csv"
        data_handler.save_data(price_df, price_filename, data_type="csv", stage="price")
        logger.info(f"Saved price data to {price_filename}")
    else:
        logger.warning("price_df is empty. Skipping price CSV save.")

    if news_df is not None and not news_df.empty:
        news_filename = f"{ticker}_news_{start_date}_to_{end_date}.csv"
        data_handler.save_data(news_df, news_filename, data_type="csv", stage="news")
        logger.info(f"Saved news data to {news_filename}")
    else:
        logger.warning("news_df is empty. Skipping news CSV save.")

if __name__ == "__main__":
    main()

Fifth file:
news_data_gatherer.py:
 import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Tuple
from src.data_aggregation.base_data_gatherer import BaseDataGatherer
from src.utils.logger import get_logger

class NewsDataGatherer(BaseDataGatherer):
    """
    Gathers news articles from Alpha Vantage over a specified date range by splitting into annual chunks.
    """
    def __init__(self, ticker: str, start_date: str, end_date: str, local_mode: bool = False) -> None:
        """
        :param ticker: Stock ticker symbol (e.g., 'TSLA').
        :param start_date: Start date in 'YYYY-MM-DD' format.
        :param end_date: End date in 'YYYY-MM-DD' format.
        :param local_mode: If True, uses local API key retrieval.
        """
        super().__init__(ticker, local_mode=local_mode)
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.base_url: str = 'https://www.alphavantage.co/query'
        self.logger = get_logger(self.__class__.__name__)

    def _generate_yearly_ranges(self) -> List[Tuple[datetime, datetime]]:
        """
        Generates a list of (start, end) datetime tuples representing yearly ranges.
        For a date range shorter than a year, it returns a single tuple.
        """
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        ranges: List[Tuple[datetime, datetime]] = []
        current_start = start_dt
        while current_start <= end_dt:
            # End the chunk on December 31 or on the actual end date, whichever is earlier.
            chunk_end = datetime(current_start.year, 12, 31)
            if chunk_end > end_dt:
                chunk_end = end_dt
            ranges.append((current_start, chunk_end))
            # Move to January 1 of the next year.
            current_start = datetime(current_start.year + 1, 1, 1)
        return ranges

    def _fetch_news_for_range(self, range_start: datetime, range_end: datetime) -> pd.DataFrame:
        """
        Fetches up to 1000 news articles for the specified datetime range.
        The time_from and time_to parameters are formatted as YYYYMMDDTHHMM.
        A 'sort=LATEST' parameter is added to retrieve the most recent articles first.
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
        self.logger.debug(f"Fetching news for range {news_start_str} to {news_end_str} with URL: {url}")
        data = self.make_api_request(url)
        if 'feed' not in data:
            self.logger.error(f"Error fetching news for range {range_start} to {range_end}: {data}")
            return pd.DataFrame()
        df = pd.DataFrame(data['feed'])
        df['Symbol'] = self.ticker
        # Convert the time_published string into a datetime object.
        df['time_published'] = pd.to_datetime(df['time_published'], format='%Y%m%dT%H%M%S', errors='coerce')
        self.logger.info(f"Fetched {len(df)} news records for {self.ticker} from {news_start_str} to {news_end_str}")
        return df

    def _fetch_news_data_in_chunks(self) -> pd.DataFrame:
        """
        Iterates over each yearly range to fetch news data and concatenates the results.
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
        Retrieves news data over the specified date range and, if possible, drops rows missing 'title' or 'summary'.
        """
        df = self._fetch_news_data_in_chunks()
        if df.empty:
            return df
        missing_cols = [col for col in ["title", "summary"] if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing columns {missing_cols}. Skipping dropna for these columns.")
        else:
            initial_count = len(df)
            df.dropna(subset=["title", "summary"], inplace=True)
            self.logger.info(f"Dropped {initial_count - len(df)} rows missing 'title' or 'summary'.")
        return df

Sixth file:
stock_price_data_gatherer.py:
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List
from src.data_aggregation.base_data_gatherer import BaseDataGatherer
from src.utils.logger import get_logger

class StockPriceDataGatherer(BaseDataGatherer):
    """
    Gathers intraday stock price data from Alpha Vantage by splitting the date range into monthly chunks.
    """
    def __init__(self, ticker: str, start_date: str, end_date: str,
                 interval: str = '1min', local_mode: bool = False) -> None:
        super().__init__(ticker, local_mode=local_mode)
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.interval: str = interval
        self.base_url: str = "https://www.alphavantage.co/query"
        self.logger = get_logger(self.__class__.__name__)

    def _generate_month_params(self) -> List[str]:
        """
        Returns a list of month parameters (e.g., "month=2022-01") for each month in the date range.
        """
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')
        date_list: List[str] = []
        current_date = start_dt
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y-%m')
            date_list.append(f"month={date_str}")
            current_date += relativedelta(months=1)
        return date_list

    def _fetch_monthly_data(self) -> pd.DataFrame:
        """
        Iterates over each month parameter to fetch and concatenate intraday price data.
        """
        date_list = self._generate_month_params()
        df_list = []
        for date_frag in date_list:
            url = (
                f"{self.base_url}?function=TIME_SERIES_INTRADAY"
                f"&symbol={self.ticker}"
                f"&interval={self.interval}"
                f"&{date_frag}"
                f"&outputsize=full"
                f"&apikey={self.api_key}"
            )
            self.logger.debug(f"Fetching stock data with URL: {url}")
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
        pricing_df = pd.concat(df_list, ignore_index=True)
        pricing_df = pricing_df[['Symbol', 'DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        pricing_df['DateTime'] = pd.to_datetime(pricing_df['DateTime'], errors='coerce')
        pricing_df.sort_values('DateTime', inplace=True)
        pricing_df.drop_duplicates(subset=['DateTime'], inplace=True)
        pricing_df.reset_index(drop=True, inplace=True)
        return pricing_df

    def run(self) -> pd.DataFrame:
        """
        Main entry point: fetches intraday price data in monthly chunks.
        """
        return self._fetch_monthly_data()


Another idea to consider and possibly implement; what if the code is easier for you to read as well.

import os
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional

# Import your existing DataAggregator and configuration.
from src.data_aggregation.data_aggregator import DataAggregator
from src.utils.logger import get_logger
from src.config import Settings

# Set up a logger for this module.
logger = get_logger("PermanentStorageMain")


class DataStorage:
    """
    DataStorage class to manage and persist data for multiple stocks over multiple time ranges.
    
    This class supports two modes:
      - 'local': Stores data as CSV files in a specified directory.
      - 'db': Stores data in an SQLite database (an open source solution).
    
    It provides functionality to save, load, update, and delete data.
    """

    def __init__(self, storage_mode: str = "local", base_path: Optional[str] = None, db_path: Optional[str] = None) -> None:
        """
        Initializes the DataStorage with the chosen storage mode and relevant paths.

        :param storage_mode: 'local' for CSV file storage or 'db' for SQLite database storage.
        :param base_path: Directory path to store CSV files when using local storage.
        :param db_path: File path for the SQLite database when using database storage.
        """
        self.storage_mode = storage_mode.lower()
        self.configs: List[Dict[str, Any]] = []  # Reserved for future multi-config support.

        if self.storage_mode == "local":
            # Use provided base_path or default to the current working directory.
            self.base_path = base_path if base_path is not None else os.getcwd()
        elif self.storage_mode == "db":
            # Use provided db_path or default to "data_storage.db".
            self.db_path = db_path if db_path is not None else "data_storage.db"
            self._init_db()
        else:
            raise ValueError("Invalid storage_mode. Must be either 'local' or 'db'.")

    def _init_db(self) -> None:
        """
        Initializes the SQLite database and creates necessary tables for storing price and news data.
        """
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Create table for price data with a composite primary key (ticker, datetime)
        create_price_table_query = """
        CREATE TABLE IF NOT EXISTS price_data (
            ticker TEXT NOT NULL,
            datetime TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, datetime)
        );
        """
        self.cursor.execute(create_price_table_query)

        # Create table for news data with a composite primary key (ticker, published_datetime, title)
        create_news_table_query = """
        CREATE TABLE IF NOT EXISTS news_data (
            ticker TEXT NOT NULL,
            published_datetime TEXT NOT NULL,
            title TEXT,
            summary TEXT,
            url TEXT,
            PRIMARY KEY (ticker, published_datetime, title)
        );
        """
        self.cursor.execute(create_news_table_query)
        self.conn.commit()

    def save_data(self, ticker: str, data: pd.DataFrame, data_type: str = "price") -> None:
        """
        Saves data for a given ticker either locally as a CSV file or in the database.

        :param ticker: Stock ticker symbol.
        :param data: DataFrame containing the data to be saved.
        :param data_type: Data type; 'price' or 'news'.
        """
        if self.storage_mode == "local":
            filename = os.path.join(self.base_path, f"{ticker}_{data_type}.csv")
            data.to_csv(filename, index=False)
        elif self.storage_mode == "db":
            if data_type == "price":
                # Insert or update each row in the price_data table.
                for _, row in data.iterrows():
                    query = """
                    INSERT OR REPLACE INTO price_data (ticker, datetime, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """
                    self.cursor.execute(query, (
                        ticker,
                        row["DateTime"],
                        row.get("Open"),
                        row.get("High"),
                        row.get("Low"),
                        row.get("Close"),
                        row.get("Volume")
                    ))
                self.conn.commit()
            elif data_type == "news":
                # Insert or update each row in the news_data table.
                for _, row in data.iterrows():
                    query = """
                    INSERT OR REPLACE INTO news_data (ticker, published_datetime, title, summary, url)
                    VALUES (?, ?, ?, ?, ?)
                    """
                    self.cursor.execute(query, (
                        ticker,
                        row.get("time_published"),
                        row.get("title"),
                        row.get("summary"),
                        row.get("url")
                    ))
                self.conn.commit()
        else:
            raise ValueError("Invalid storage_mode. Must be either 'local' or 'db'.")

    def load_data(self, ticker: str, start_date: str, end_date: str, data_type: str = "price") -> pd.DataFrame:
        """
        Loads data for a given ticker and date range from storage.

        :param ticker: Stock ticker symbol.
        :param start_date: Start date (format 'YYYY-MM-DD').
        :param end_date: End date (format 'YYYY-MM-DD').
        :param data_type: Data type; 'price' or 'news'.
        :return: DataFrame containing the filtered data.
        """
        if self.storage_mode == "local":
            filename = os.path.join(self.base_path, f"{ticker}_{data_type}.csv")
            if os.path.exists(filename):
                data = pd.read_csv(filename)
                if data_type == "price":
                    data["DateTime"] = pd.to_datetime(data["DateTime"])
                    mask = (data["DateTime"] >= start_date) & (data["DateTime"] <= end_date)
                    return data.loc[mask]
                elif data_type == "news":
                    data["time_published"] = pd.to_datetime(data["time_published"])
                    mask = (data["time_published"] >= start_date) & (data["time_published"] <= end_date)
                    return data.loc[mask]
                else:
                    return data
            else:
                return pd.DataFrame()
        elif self.storage_mode == "db":
            if data_type == "price":
                query = """
                SELECT * FROM price_data
                WHERE ticker = ?
                AND datetime BETWEEN ? AND ?
                ORDER BY datetime
                """
                self.cursor.execute(query, (ticker, start_date, end_date))
                rows = self.cursor.fetchall()
                if rows:
                    columns = ["ticker", "DateTime", "Open", "High", "Low", "Close", "Volume"]
                    data = pd.DataFrame(rows, columns=columns)
                    data["DateTime"] = pd.to_datetime(data["DateTime"])
                    return data
                else:
                    return pd.DataFrame()
            elif data_type == "news":
                query = """
                SELECT * FROM news_data
                WHERE ticker = ?
                AND published_datetime BETWEEN ? AND ?
                ORDER BY published_datetime
                """
                self.cursor.execute(query, (ticker, start_date, end_date))
                rows = self.cursor.fetchall()
                if rows:
                    columns = ["ticker", "published_datetime", "title", "summary", "url"]
                    data = pd.DataFrame(rows, columns=columns)
                    data["published_datetime"] = pd.to_datetime(data["published_datetime"])
                    return data
                else:
                    return pd.DataFrame()
        else:
            raise ValueError("Invalid storage_mode. Must be either 'local' or 'db'.")

    def delete_data(self, ticker: str, data_type: str = "price") -> None:
        """
        Deletes stored data for a given ticker.

        :param ticker: Stock ticker symbol.
        :param data_type: Data type to delete; 'price' or 'news'.
        """
        if self.storage_mode == "local":
            filename = os.path.join(self.base_path, f"{ticker}_{data_type}.csv")
            if os.path.exists(filename):
                os.remove(filename)
        elif self.storage_mode == "db":
            if data_type == "price":
                query = "DELETE FROM price_data WHERE ticker = ?"
                self.cursor.execute(query, (ticker,))
            elif data_type == "news":
                query = "DELETE FROM news_data WHERE ticker = ?"
                self.cursor.execute(query, (ticker,))
            self.conn.commit()
        else:
            raise ValueError("Invalid storage_mode. Must be either 'local' or 'db'.")

    def update_data(self, ticker: str, data: pd.DataFrame, data_type: str = "price") -> None:
        """
        Updates stored data for a given ticker by merging new data with existing records.

        :param ticker: Stock ticker symbol.
        :param data: DataFrame containing the new data.
        :param data_type: Data type to update; 'price' or 'news'.
        """
        # Load all existing data for the ticker (using a wide date range)
        existing_data = self.load_data(ticker, "1900-01-01", "2100-01-01", data_type)
        combined_data = pd.concat([existing_data, data]).drop_duplicates()

        if self.storage_mode == "local":
            filename = os.path.join(self.base_path, f"{ticker}_{data_type}.csv")
            combined_data.to_csv(filename, index=False)
        elif self.storage_mode == "db":
            # Delete existing records and reinsert the combined data.
            self.delete_data(ticker, data_type)
            self.save_data(ticker, combined_data, data_type)
        else:
            raise ValueError("Invalid storage_mode. Must be either 'local' or 'db'.")


def main() -> None:
    """
    Main function to create a more permanent storage solution for data API calls.

    This function:
      1. Loads configuration parameters (ticker, start/end dates, etc.) via Settings.
      2. Sets up DataStorage (either local CSV storage or an SQLite database).
      3. Attempts to load stored price and news data for the given ticker and date range.
      4. If stored data is missing, it calls DataAggregator to fetch fresh data via API calls.
      5. Stores the newly fetched data permanently so that subsequent runs can load the data directly.
    """
    # Load configuration via the Settings model.
    settings = Settings()  # Assumes attributes: ticker, start_date, end_date, interval, local_mode
    ticker = settings.ticker
    start_date = settings.start_date
    end_date = settings.end_date
    interval = settings.interval
    local_mode = settings.local_mode

    # Set up DataStorage.
    # Change 'local' to 'db' to use an SQLite database instead.
    storage_mode = "local"
    if storage_mode == "local":
        permanent_storage_path = os.path.join(os.getcwd(), "permanent_storage")
        os.makedirs(permanent_storage_path, exist_ok=True)
        data_storage = DataStorage(storage_mode=storage_mode, base_path=permanent_storage_path)
    else:
        data_storage = DataStorage(storage_mode=storage_mode)

    # Attempt to load stored data for the specified ticker and date range.
    stored_price_df = data_storage.load_data(ticker, start_date, end_date, data_type="price")
    stored_news_df = data_storage.load_data(ticker, start_date, end_date, data_type="news")

    if stored_price_df.empty or stored_news_df.empty:
        logger.info(f"No stored data found for {ticker} from {start_date} to {end_date}.")
        logger.info("Calling DataAggregator to fetch fresh data from the API...")
        # Instantiate DataAggregator to fetch data via API calls.
        aggregator = DataAggregator(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            local_mode=local_mode
        )
        price_df, news_df = aggregator.aggregate_data()
        # Save the fetched data permanently.
        data_storage.save_data(ticker, price_df, data_type="price")
        data_storage.save_data(ticker, news_df, data_type="news")
        logger.info("Data has been fetched and stored permanently.")
    else:
        logger.info("Permanent stored data found. Using stored data to avoid redundant API calls.")
        price_df = stored_price_df
        news_df = stored_news_df

    # Verification: Print a summary of the loaded data.
    print(f"{ticker} price data records: {len(price_df)}")
    print(f"{ticker} news data records: {len(news_df)}")


if __name__ == "__main__":
    main()
