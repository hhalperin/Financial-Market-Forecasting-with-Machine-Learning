# Stock Data & News Aggregation and Processing Pipeline

This project provides a modular Python codebase for aggregating, processing, and analyzing stock market data along with related news articles. The system integrates several components including:

- **Data Aggregation:** Fetches stock price and news data via APIs (e.g., Alpha Vantage) and aggregates the data.
- **Data Processing:** Cleans and merges the aggregated data, computes technical indicators, performs sentiment analysis, and generates embeddings from news text.
- **Permanent Storage:** Provides mechanisms to cache data locally or in an SQLite database to avoid repeated API calls during development.
- **Performance Monitoring:** Automatically identifies performance bottlenecks using a decorator that logs function execution times.

## Features

- **Configurable Parameters:** All key parameters are configurable via a centralized `config.py` file.
- **Data Embedding:** Uses Hugging Face models to generate text embeddings. Each text column is replaced by a single column containing the full embedding vector.
- **Sentiment Analysis:** Leverages state-of-the-art models to perform sentiment analysis on news articles.
- **Technical Analysis:** Calculates market indicators (RSI, MACD, ROC, etc.) using libraries such as TA-Lib.
- **Time Horizon Management:** Generates combinations of time horizons for model training.
- **Performance Logging:** Automatically logs potential performance bottlenecks with detailed timing information.
- **Automated Linting:** GitHub Actions are configured to lint the codebase on every push.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [TA-Lib](https://mrjbq7.github.io/ta-lib/) (for technical indicators)
- A virtual environment is recommended

### Installation

1. **Clone the repository:**

   ```git clone https://github.com/yourusername/your-repo-name.git```
   cd your-repo-name

Create and activate a virtual environment:


```python -m venv venv````
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies:


```pip install -r requirements.txt```
Set up your environment variables:

Create a .env file in the root directory with the required API keys and configurations (refer to config.py for details).

Usage
Data Aggregation & Processing:

Run the main scripts to fetch, process, and store data:

```
python src/data_aggregation/main.py
python src/data_processing/main.py
```

Performance Monitoring:

Key functions are decorated with @profile_time. Check your logs for any warnings regarding performance bottlenecks.

Continuous Integration
This project uses GitHub Actions for automated linting (see .github/workflows/python-lint.yml).

Contributing
Contributions are welcome! Please ensure your changes follow the coding style and that all tests pass.

License
MIT License