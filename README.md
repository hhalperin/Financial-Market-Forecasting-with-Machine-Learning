Financial Market Forecasting with Machine Learning

This project aims to forecast financial market trends using machine learning techniques. It involves gathering data from various sources (e.g., news and stock prices), processing and preparing the data, training machine learning models to predict market trends, and (in progress) implementing trading strategies based on those predictions.

The project is organized into several directories, each handling a specific part of the workflow, from data aggregation to model training and trading.

Table of Contents

Installation

Usage

Directory Structure

quant-h2_website

src

data

data_aggregation

data_processing

models

trading (WIP)

utils

Contributing

License

Installation

To set up the project, ensure you have Python installed. Then, install the required dependencies:

pip install -r src/requirements.txt

You may also need to configure API keys or other settings in src/config.py before running the scripts.

Usage

The project is modular, with each component having its own entry point:

Data Aggregation: Run src/data_aggregation/main.py to collect data from various sources.

Data Processing: Run src/data_processing/main.py to process and prepare the data for modeling.

Model Training: Run src/models/main.py to train and evaluate the machine learning models.

Trading (WIP): The trading functionality is under development; check src/trading/main.py for current capabilities.

Ensure src/config.py is properly configured with necessary settings (e.g., API keys, file paths) before execution.

Directory Structure

Here’s an overview of the project’s structure, with detailed breakdowns below:

quant-h2_website/: Files for deploying a project-related website.

src/: Main source code directory containing all core functionality.

data/: Stores datasets and trained models.

data_aggregation/: Gathers data from external sources.

data_processing/: Prepares data for modeling.

models/: Defines and manages machine learning models.

trading/: In-progress trading functionality.

utils/: Shared utility functions.

quant-h2_website

This directory contains files for deploying a website related to the project.

terraform/: Infrastructure-as-code files for website deployment.

files/: Contains HTML and CSS files for the website frontend.

Terraform files: Configuration files for managing infrastructure (e.g., AWS, GCP) using Terraform.

src

The root of the source code, containing configuration and requirements:

__init__.py: Marks src/ as a Python package.

config.py: Central configuration file for the project (e.g., API keys, paths).

requirements.txt: Lists Python dependencies.

data

Stores data and models used in the project.

models/: Saved machine learning models.

news/: Raw news data collected for sentiment analysis.

numeric/: Numeric data (e.g., financial metrics).

preprocessed/: Processed data ready for modeling.

price/: Stock price data.

data_aggregation

This module is responsible for gathering and aggregating stock price and news data from the Alpha Vantage API. It provides a suite of classes to fetch data concurrently using threading for improved performance, manage API interactions with retry logic, and store data persistently in either local CSV files or an SQLite database.

__init__.py: Initializes the data_aggregation package and exports key classes.

base_data_gatherer.py: Provides the foundation for data gathering operations.

data_aggregator.py: Manages the aggregation of stock price and news data.

news_data_gatherer.py: Fetches news articles from Alpha Vantage.

stock_price_data_gatherer.py: Retrieves intraday stock price data from Alpha Vantage.

data_storage.py: Manages persistent storage of aggregated data.

main.py: The entry point for the data aggregation process.

data_processing

Processes and prepares data for use in machine learning models.

__init__.py: Marks this as a Python package.

data_processing.py: Core data processing functions (e.g., cleaning, normalization).

data_embedder.py: Converts data (e.g., text) into embeddings or numeric representations.

main.py: Entry point to process data.

sentiment_processor.py: Analyzes sentiment from news or text data.

market_analyzer.py: Extracts insights from market data.

time_horizon_manager.py: Aligns data across different forecasting timeframes.

models

Defines and manages the machine learning models for market forecasting.

__init__.py: Marks this as a Python package.

cpu_optimization.py: Optimizes model performance for CPU execution.

stock_predictor.py: Core model for predicting stock prices or trends.

model_manager.py: Handles model training, saving, and loading.

main.py: Entry point for training and evaluating models.

model_analysis.py: Evaluates model performance (e.g., metrics, visualizations).

model_pipeline.py: Defines the end-to-end modeling pipeline.

model_summary.py: Generates summaries of model architecture or results.

configuration.py: Model-specific configuration settings.

Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub for bug fixes, improvements, or new features.

License

This project is licensed under the MIT License.

#### data_aggregation

This module is responsible for gathering and aggregating stock price and news data from the Alpha Vantage API. It provides a suite of classes to fetch data concurrently using threading for improved performance, manage API interactions with retry logic, and store data persistently in either local CSV files or an SQLite database.

<details>
<summary>Files in data_aggregation</summary>

- **`__init__.py`**: Initializes the `data_aggregation` package and exports key classes:
  - `DataAggregator`
  - `NewsDataGatherer`
  - `StockPriceDataGatherer`
  - `BaseDataGatherer`

- **`base_data_gatherer.py`**: Provides the foundation for data gathering operations.
  - **Classes/Functions**:
    - `BaseDataGatherer`: A base class for specialized data gatherers, handling API key retrieval (from environment variables or AWS Secrets Manager) and HTTP requests with retry logic using a persistent `requests.Session`.
    - `get_cached_api_key`: Caches and retrieves API keys to avoid redundant lookups.
  - **Key Features**: Supports local and production modes, uses `tenacity` for retries, and logs operations.

- **`data_aggregator.py`**: Manages the aggregation of stock price and news data.
  - **Class**: `DataAggregator`
    - Aggregates data concurrently using `ThreadPoolExecutor` for performance.
    - Coordinates `StockPriceDataGatherer` and `NewsDataGatherer` to fetch data.
  - **Key Features**: Supports custom intervals (e.g., "1min") and optional data handlers for further processing.

- **`news_data_gatherer.py`**: Fetches news articles from Alpha Vantage.
  - **Class**: `NewsDataGatherer` (inherits from `BaseDataGatherer`)
    - Splits date ranges into yearly chunks to manage API limits.
    - Returns a `pandas.DataFrame` with news data (e.g., title, summary, URL).
  - **Key Features**: Handles large date ranges, deduplicates articles, and drops incomplete records.

- **`stock_price_data_gatherer.py`**: Retrieves intraday stock price data from Alpha Vantage.
  - **Class**: `StockPriceDataGatherer` (inherits from `BaseDataGatherer`)
    - Splits date ranges into monthly chunks for efficient API calls.
    - Returns a `pandas.DataFrame` with price data (e.g., Open, High, Low, Close, Volume).
  - **Key Features**: Supports configurable intervals, ensures data consistency, and suggests Parquet for large datasets.

- **`data_storage.py`**: Manages persistent storage of aggregated data.
  - **Class**: `DataStorage`
    - Supports two modes: "local" (CSV files) and "db" (SQLite database).
    - Provides methods: `save_data`, `load_data`, `update_data`, `delete_data`.
  - **Key Features**: Initializes database tables, handles data merging, and supports flexible storage paths.
  - **Usage in Main**: Integrates with `DataAggregator` to avoid redundant API calls by loading stored data first.

- **`main.py`**: The entry point for the data aggregation process.
  - **Functions**:
    - `main`: Orchestrates configuration loading, data handler setup, aggregation, and storage.
    - `load_config`: Loads settings from `Settings` (e.g., ticker, dates, mode).
    - `setup_data_handler`: Configures `DataHandler` for local or S3 storage.
    - `setup_aggregator`: Initializes `DataAggregator`.
  - **Key Features**: Supports local and S3 storage modes, logs progress, and saves data as CSV files.

</details>

**Usage**: To run the data aggregation process, execute `src/data_aggregation/main.py`. Ensure `src/config.py` is configured with required settings (e.g., API keys, ticker, date ranges). The script checks for existing data in storage and fetches fresh data from the API if needed.

**Dependencies**: 
- Standard libraries: `os`, `json`, `time`
- External libraries: `pandas`, `requests`, `boto3`, `sqlite3`, `tenacity`, `python-dotenv`

**Data Source**: For more details on the API, refer to the [Alpha Vantage API documentation](https://www.alphavantage.co/documentation/).