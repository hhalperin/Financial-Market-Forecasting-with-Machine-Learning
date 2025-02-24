# src/trading/trading_config.py
from src.config import Settings  # Import base settings

class TradingSettings(Settings):  # Inherit from Settings
    # -------------------------------
    # Trading-Specific Configuration (Overrides)
    # -------------------------------
    initial_capital: float = 100000.0  # Starting balance for backtesting
    commission: float = 0.001          # Commission per trade
    slippage: float = 0.0005           # Slippage (fraction of price)

    # Backtesting Parameters (for historical data)
    backtest_ticker: str = "NVDA"
    backtest_start_date: str = "2025-01-02"
    backtest_end_date: str = "2025-02-01"
    interval: str = "1min"
    outputsize: str = "full"

    # Live Mode Settings
    live_mode: bool = False          # Set to True for live data
    live_days: int = 1               # How many days of data to use in live mode

    # Model Storage Paths (Ensures No Conflict With General Storage Paths)
    best_models_dir: str = "./src/data/models/best_models"
    goated_models_dir: str = "./src/data/models/goated_models"

    # Trading-Specific Visualization Parameters
    equity_curve_title: str = "Equity Curve"
    drawdown_title: str = "Drawdowns"
    trade_histogram_title: str = "Trade PnL Histogram"
    histogram_bins: int = 20

    # Override conflicting storage paths to be trading-specific
    data_storage_path: str = "./src/data/trading"
    permanent_storage_path: str = "./permanent_storage/trading"

    # Ensure that trading-specific environment variables load correctly
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global instance
trading_settings = TradingSettings()
