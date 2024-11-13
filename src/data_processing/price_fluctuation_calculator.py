import pandas as pd
from logger import get_logger

class PriceFluctuationCalculator:
    """
    Calculates price fluctuations for specified time periods.
    """

    def __init__(self, data_df, time_horizons):
        self.data_df = data_df.copy()
        self.time_horizons = time_horizons
        self.logger = get_logger(self.__class__.__name__)

    def calculate_fluctuations(self):
        """
        Calculates price changes and percentage changes for each time horizon.
        """
        self.data_df.set_index('DateTime', inplace=True)
        self.data_df['Close'] = pd.to_numeric(self.data_df['Close'], errors='coerce')

        for config in self.time_horizons:
            horizon_name = config['target_name']
            time_horizon_minutes = int(config['time_horizon'].total_seconds() // 60)
            
            # Calculate price change and percentage change for the given time horizon
            shifted_close = self.data_df['Close'].shift(-time_horizon_minutes)
            self.data_df[f'{horizon_name}_change'] = shifted_close - self.data_df['Close']
            self.data_df[f'{horizon_name}_percentage_change'] = ((shifted_close - self.data_df['Close']) / self.data_df['Close']) * 100

        self.data_df.reset_index(inplace=True)
        self.logger.info("Calculated price fluctuations.")
        return self.data_df
