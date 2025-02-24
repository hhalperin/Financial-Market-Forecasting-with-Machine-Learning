# Financial Market Forecasting using Machine Learning

## Overview

This project is a comprehensive system for collecting, processing, modeling, and eventually simulating live trading based on stock market data. The system is designed to:

- **Aggregate Data:** Fetch intraday stock price data and related news articles from the Alpha Vantage API.
- **Process Data:** Clean, merge, and transform the aggregated data—generating market features, sentiment scores, and text embeddings.
- **Model Training & Evaluation:** Train multiple neural network models to predict stock price fluctuations using various time horizons.
- **Trading Simulation:** Aggregate real-time data and, using pre-trained models along with configurable trading rules, simulate trades and test strategy performance.

## Directory Structure

```bash
project-root/
│
├── data_aggregation/
│   ├── __init__.py
│   ├── base_data_gatherer.py
│   ├── data_aggregator.py
│   ├── news_data_gatherer.py
│   └── stock_price_data_gatherer.py
│
├── data_processing/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── data_embedder.py
│   ├── sentiment_processor.py
│   ├── market_analyzer.py
│   └── time_horizon_manager.py
│
├── models/
│   ├── __init__.py
│   ├── stock_predictor.py
│   ├── model_manager.py
│   ├── model_analysis.py
│   ├── model_pipeline.py
│   └── configuration.py
│
├── trading/
│   ├── __init__.py
│   ├── trading_config.py   # Trading-specific configuration (extends base settings)
│   ├── live_data_collector.py  # Continuously polls for live data
│   ├── trading_engine.py        # Orchestrates live data processing and trading decisions
│   ├── trading_rules.py         # Contains rule-based trading logic
│   ├── trade_simulator.py       # Simulates trades based on model predictions
│   └── dashboard.py             # (Future) Secure live dashboard for monitoring trades
│
└── utils/
    ├── data_handler.py
    ├── data_loader.py
    ├── error_handler.py
    ├── logger.py
    └── performance_monitor.py
```

## Configuration

The system uses a centralized configuration file (`src/config.py`) based on Pydantic’s `BaseSettings` to load parameters from environment variables (via a `.env` file). For trading-specific settings, a new file—`trading/trading_config.py`—extends these settings to include parameters such as:

- **Initial Capital:** Starting balance for backtesting.
- **Commission & Slippage:** Costs per trade.
- **Live Mode Toggles:** Enable or disable live data collection.
- **Live Data Days:** Number of days of live data to use.
- **Storage Paths:** Trading-specific paths to avoid conflicts with other modules.
- **Visualization Parameters:** Titles for equity curve, drawdowns, and trade histograms.

Make sure to create a `.env` file in the project root with the necessary API keys and any custom settings.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   (Ensure your requirements.txt includes packages such as pydantic_settings, requests, boto3, tenacity, pandas, numpy, talib, torch, transformers, scikit-learn, etc.)

## Running the System

### 1. Data Aggregation

To fetch historical price and news data:

```bash
python src/main.py
```

This script loads configuration, aggregates data using modules in `data_aggregation/`, and stores the results locally or in a database.

### 2. Data Processing

To clean, merge, and enrich the aggregated data:

```bash
python src/data_processing/main.py
```

This step produces preprocessed and numeric data for model training.

### 3. Model Training & Evaluation

To train your stock prediction models across various time horizons:

```bash
python src/models/main.py
```

This module trains multiple models, evaluates them, and optionally performs hyperparameter optimization using Optuna.

### 4. Trading Simulation (Live Trading)

**Live Data Collector**

The new `live_data_collector.py` module (located in `trading/`) wraps the data aggregation code in a continuous loop. It is designed to:

- **Before Market Open:** Poll and fetch all after-hours data at market open.
- **During Market Hours:** Poll live data every minute (trading occurs only during market hours).

**Starting the Trading Engine**

Once the live data collector is running, the trading engine (in `trading/trading_engine.py`) processes the new data, applies your trading rules (defined in `trading/trading_rules.py`), and simulates trade executions via the trade simulator (`trading/trade_simulator.py`).

To start the live trading simulation, set the `live_mode` boolean to `True` in your `trading/trading_config.py` and run:

```bash
python src/trading/live_data_collector.py
```

(Depending on your final integration, the trading engine might be invoked from within the live data collector or as a separate process.)

### 5. Dashboard (Future)

A secure, password-protected dashboard (using Flask, Django, or Streamlit) is planned to provide a live view of:

- Stock price charts.
- Expected price fluctuation graphs.
- Trade logs and performance metrics.

## Future Enhancements

- **Data Storage:** Migrate from local CSV/SQLite to cloud-based databases as data volume increases.
- **Advanced Trading Rules:** Start with simple threshold-based rules and evolve toward more complex, adaptive strategies (e.g., dynamic stop-loss or neural network-based parameter prediction).
- **Dashboard Integration:** Build a secure web dashboard for real-time monitoring and control.
- **Automated Model Updates:** Implement automated retraining using the most recent data during live trading.
- **Enhanced Error Handling & Alerts:** Improve error handling and logging, and add real-time alerts for critical system failures.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to help improve the project.

## License

This project is licensed under the MIT License.

For any questions or further clarifications, please open an issue in the repository or contact the maintainers.

