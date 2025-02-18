# SEMPER: Stock Embedding Models for Predicting Expected Returns

SEMPER stands for **Stock Embedding Models for Predicting Expected Returns**. This project provides a modular Python codebase designed to aggregate, process, and analyze stock market data and related news articles. The processed data is used to train and run machine learning models that predict expected returns, enabling simulated trading based on these predictions.

---

## Overview

The project is organized into several key components:

- **Data Aggregation:**  
  Retrieves raw intraday stock price data and news articles via APIs (e.g., Alpha Vantage) using a concurrent fetching mechanism. Raw data is cached locally (or in a database) to avoid redundant API calls during development.

- **Data Processing:**  
  Cleans and merges raw data, calculates technical indicators (RSI, MACD, ROC, etc.), and performs sentiment analysis using advanced NLP models. Additionally, it generates composite text embeddings using Hugging Face transformer models, combining multiple text fields into a single embedding vector. This processed numeric data is used for model training and prediction.

- **Permanent Storage:**  
  At each stage, data is saved as CSV (for processed data) or as a NumPy `.npy` file (for numeric features) so that subsequent runs can load saved data quickly, which is especially useful during testing and development.

- **Trading Simulation:**  
  A trading engine loads a pre-trained model (using “goated” model checkpoints) and uses the processed numeric data to predict expected price fluctuations. Based on simple threshold-based trading rules, the engine simulates trade decisions (BUY, SELL, or HOLD) and logs every decision with its rationale.

- **Performance Monitoring:**  
  Critical functions throughout the pipeline are decorated with a performance monitor (`@profile_time`) that logs execution times and warns when a function exceeds a specified threshold.

- **Continuous Integration:**  
  The project will be set up with GitHub Actions for automated linting and testing on every push, ensuring code quality and consistency.

---

## Features

- **Modular Architecture:**  
  The codebase is divided into distinct modules for data aggregation, processing, model training, and trading simulation.

- **Configurable Settings:**  
  All configurable parameters (API keys, storage modes, model hyperparameters, trading thresholds, etc.) are centralized in `config.py` and `trading_config.py`.

- **Composite Text Embeddings:**  
  News text from multiple fields (e.g., title, authors, summary) is combined into a single composite embedding, reducing dimensionality and simplifying model input.

- **Advanced Sentiment Analysis:**  
  Uses state-of-the-art NLP models (e.g., FinBERT) with recency weighting to compute sentiment scores from news articles.

- **Technical Indicators:**  
  Computes market technical indicators such as RSI, MACD, and ROC using TA-Lib.

- **Time Horizon Management:**  
  Generates multiple time horizon combinations to support robust model training and analysis.

- **Robust Data Caching:**  
  Saves data at each processing stage to CSV or `.npy` formats, allowing for rapid reloading during development and testing.

- **Simple Trading Rules:**  
  Implements threshold-based trading rules to decide trade actions (BUY, SELL, or HOLD) based on predicted expected returns.

- **Extensive Logging:**  
  Detailed logging is built in at every stage—from data aggregation to prediction—making debugging and performance analysis easier.

- **Performance Monitoring:**  
  Key functions are instrumented with a timing decorator that logs execution times, helping identify bottlenecks.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [TA-Lib](https://mrjbq7.github.io/ta-lib/) for technical indicator calculations
- A virtual environment is recommended

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
