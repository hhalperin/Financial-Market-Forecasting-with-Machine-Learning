import time
from typing import Tuple
import pandas as pd
from datetime import datetime, timedelta
from logger import get_logger
from .news_data_gatherer import NewsDataGatherer
from .stock_price_data_gatherer import StockPriceDataGatherer
from utils.data_handler import DataHandler
from dateutil.relativedelta import relativedelta

logger = get_logger('DataAggregation')

def validate_date(start_date_str, end_date_str):
    def correct_date(date_str):
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            # Correct the date if the day is out of range for the month
            year, month, day = map(int, date_str.split('-'))
            last_valid_day = (datetime(year, month, 1) + relativedelta(months=1) - relativedelta(days=1)).day
            corrected_day = min(day, last_valid_day)
            return datetime(year, month, corrected_day)
    
    corrected_start_date = correct_date(start_date_str).strftime('%Y-%m-%d')
    corrected_end_date = correct_date(end_date_str).strftime('%Y-%m-%d')

    return corrected_start_date, corrected_end_date

def generate_monthly_date_ranges(start_date: str, end_date: str) -> list:
    """
    Generates a list of monthly date ranges between start_date and end_date.

    Args:
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.

    Returns:
        List of str: Each element is a month in "YYYY-MM" format.
    """
    # Validate and correct dates if necessary
    start_date, end_date = validate_date(start_date, end_date)

    # Convert validated strings to datetime objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    current_date = start_date
    date_ranges = []

    while current_date <= end_date:
        month_str = current_date.strftime("%Y-%m")
        date_ranges.append(month_str)
        # Move to the next month
        next_month = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1)
        current_date = next_month

    return date_ranges

def aggregate_data(ticker: str, start_date: str, end_date: str, interval: str = '1min', outputsize: str = 'full') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregates price and news data for the given ticker and date range.

    Args:
        ticker (str): The stock ticker symbol.
        start_date (str): The start date for the data.
        end_date (str): The end date for the data.
        interval (str): The interval for price data ('1d', '1h', etc.).
        outputsize (str): The output size, either 'compact' or 'full'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Combined Price DataFrame and News DataFrame.
    """
    total_start_time = time.time()
    
    # Generate monthly date ranges
    monthly_date_ranges = generate_monthly_date_ranges(start_date, end_date)

    # Initialize an empty DataFrame to hold all monthly price data
    all_price_data = []

    # Fetch Price Data for each month
    for month in monthly_date_ranges:
        logger.info(f"Fetching price data for {ticker} for the month: {month}...")
        
        # Use StockPriceDataGatherer to gather data for each specific month
        price_gatherer = StockPriceDataGatherer(ticker=ticker, interval=interval, outputsize=outputsize, month=month)
        monthly_price_df = price_gatherer.run()
        
        # Append monthly DataFrame to the list if it's not empty
        if monthly_price_df is not None and not monthly_price_df.empty:
            all_price_data.append(monthly_price_df)
        else:
            logger.warning(f"No data fetched for {ticker} for the month: {month}.")

        # Respect rate limits by sleeping for a while after each API call
        time.sleep(12)

    # Combine all monthly dataframes
    if all_price_data:
        price_df = pd.concat(all_price_data, ignore_index=True)
        logger.info(f"Combined price data shape: {price_df.shape}")
    else:
        price_df = pd.DataFrame()
        logger.error(f"No price data was fetched for the entire date range: {start_date} to {end_date}.")

    # Fetch News Data
    logger.info(f"Fetching news data for {ticker} from {start_date} to {end_date}...")
    news_gatherer = NewsDataGatherer(ticker=ticker, start_date=start_date, end_date=end_date)
    news_df = news_gatherer.run()

    # Log completion time
    aggregation_end_time = time.time()
    logger.info(f"Data aggregation completed in {aggregation_end_time - total_start_time:.2f} seconds.")

    return price_df, news_df
