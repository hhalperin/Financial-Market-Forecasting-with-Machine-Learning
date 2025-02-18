"""
live_data_collector.py

This module continuously collects live stock data by wrapping the existing data aggregation logic.
- At market open, it polls all after-hours data (from previous market close to today’s open).
- During market hours (9:30 AM to 4:00 PM), it polls live data every minute.
- Logging is provided for every polling cycle for analysis.
"""

import time
import datetime
from src.data_aggregation.data_aggregator import DataAggregator
from src.utils.logger import get_logger

class LiveDataCollector:
    def __init__(self, ticker: str, local_mode: bool = True):
        """
        Initializes the LiveDataCollector.
        
        :param ticker: Stock ticker symbol.
        :param local_mode: Boolean flag for local mode.
        """
        self.ticker = ticker
        self.local_mode = local_mode
        self.logger = get_logger(self.__class__.__name__)
        
        # Define market hours (assumed Eastern Time; adjust as needed)
        self.market_open = datetime.time(9, 30)
        self.market_close = datetime.time(16, 0)
        
        # Polling interval during market hours (in seconds)
        self.poll_interval_seconds = 60  # 1 minute
        
        # Flag to ensure after-hours data is polled once at market open
        self.after_hours_polled = False

    def is_market_open(self) -> bool:
        """Checks if current time is within market hours."""
        now = datetime.datetime.now().time()
        return self.market_open <= now <= self.market_close

    def run(self) -> None:
        """Starts the continuous live data collection loop."""
        self.logger.info("Starting Live Data Collector loop.")
        while True:
            now = datetime.datetime.now()
            
            if not self.is_market_open():
                self.logger.info("Market is closed. Waiting until market opens.")
                time.sleep(300)  # Sleep for 5 minutes
                continue
            
            # At market open, poll after-hours data once.
            market_open_datetime = datetime.datetime.combine(now.date(), self.market_open)
            one_minute_after_open = (market_open_datetime + datetime.timedelta(minutes=1)).time()
            if now.time() < one_minute_after_open and not self.after_hours_polled:
                self.logger.info("Market just opened. Polling after-hours data.")
                previous_day = now.date() - datetime.timedelta(days=1)
                start_datetime = datetime.datetime.combine(previous_day, self.market_close)
                end_datetime = market_open_datetime
                self.logger.info(f"Polling after-hours data from {start_datetime} to {end_datetime}.")
                try:
                    aggregator = DataAggregator(
                        ticker=self.ticker,
                        start_date=start_datetime.strftime("%Y-%m-%d"),
                        end_date=end_datetime.strftime("%Y-%m-%d"),
                        interval="1min",
                        local_mode=self.local_mode
                    )
                    price_df, news_df = aggregator.aggregate_data()
                    self.logger.info(f"After-hours data: {len(price_df)} price records, {len(news_df)} news records.")
                except Exception as e:
                    self.logger.error(f"Error polling after-hours data: {e}")
                self.after_hours_polled = True

            # Poll live data during market hours.
            market_start = market_open_datetime
            now_str = now.strftime("%Y-%m-%d")
            start_str = market_start.strftime("%Y-%m-%d")
            self.logger.info(f"Polling live data from market open ({start_str}) to current time ({now.strftime('%H:%M:%S')}).")
            try:
                aggregator = DataAggregator(
                    ticker=self.ticker,
                    start_date=start_str,
                    end_date=now_str,
                    interval="1min",
                    local_mode=self.local_mode
                )
                price_df, news_df = aggregator.aggregate_data()
                self.logger.info(f"Live data: {len(price_df)} price records, {len(news_df)} news records.")
                # Optionally trigger downstream processing here.
            except Exception as e:
                self.logger.error(f"Error polling live data: {e}")

            self.logger.info(f"Sleeping for {self.poll_interval_seconds} seconds before next poll.")
            time.sleep(self.poll_interval_seconds)
            
            # Reset after_hours_polled flag for next day.
            if now.time() > self.market_close:
                self.after_hours_polled = False

if __name__ == "__main__":
    # Example usage: Collect live data for NVDA in local mode.
    collector = LiveDataCollector(ticker="NVDA", local_mode=True)
    collector.run()
