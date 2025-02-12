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
