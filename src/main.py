import os
import time
import pandas as pd
import numpy as np
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from data_aggregation.stock_price_data_gatherer import StockPriceDataGatherer
from data_aggregation.news_data_gatherer import NewsDataGatherer
from data_processing.price_fluctuation_calculator import PriceFluctuationCalculator
from data_processing.technical_indicator_calculator import TechnicalIndicatorCalculator
from data_processing.sentiment_analyzer import SentimentAnalyzer
from data_processing.data_preprocessor import DataPreprocessor
from data_processing.data_embedder import DataEmbedder 
from models.stock_predictor import StockPredictor
from models.model_trainer import ModelTrainer
from models.model_evaluator import ModelEvaluator
from models.model_visualizer import ModelVisualizer
from utils.logger import get_logger
from sklearn.model_selection import train_test_split

def main():
    logger = get_logger('Main')
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'

    total_start_time = time.time()

    # Define the data directories
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    # Define subdirectories
    news_dir = os.path.join(data_dir, 'news')
    pricing_dir = os.path.join(data_dir, 'pricing')
    technical_indicators_dir = os.path.join(data_dir, 'technical_indicators')
    sentiment_dir = os.path.join(data_dir, 'sentiment_analyzer')
    combined_dir = os.path.join(data_dir, 'combined')
    embedded_dir = os.path.join(data_dir, 'embedded')

    # Ensure subdirectories exist
    for directory in [news_dir, pricing_dir, technical_indicators_dir, sentiment_dir, combined_dir, embedded_dir]:
        os.makedirs(directory, exist_ok=True)

    # Prepare filename components
    date_range = f"{start_date}_to_{end_date}"

    # Check if files already exist to skip unnecessary API calls
    # Price Data
    price_filename = f"{ticker}_{date_range}_pricing.csv"
    price_filepath = os.path.join(pricing_dir, price_filename)

    if os.path.exists(price_filepath):
        logger.info(f"Price data already exists at {price_filepath}. Loading data...")
        price_df = pd.read_csv(price_filepath)
    else:
        logger.info("Fetching price data...")
        price_gatherer = StockPriceDataGatherer(ticker)
        price_df = price_gatherer.run()
        # Save price_df
        price_df.to_csv(price_filepath, index=False)
        logger.info(f"Price data saved to {price_filepath}")

    # News Data
    news_filename = f"{ticker}_{date_range}_news.csv"
    news_filepath = os.path.join(news_dir, news_filename)

    if os.path.exists(news_filepath):
        logger.info(f"News data already exists at {news_filepath}. Loading data...")
        news_df = pd.read_csv(news_filepath)
    else:
        logger.info("Fetching news data...")
        news_gatherer = NewsDataGatherer(ticker, start_date, end_date)
        news_df = news_gatherer.run()
        # Save news_df
        news_df.to_csv(news_filepath, index=False)
        logger.info(f"News data saved to {news_filepath}")

    aggregation_end_time = time.time()
    logger.info(f"Data aggregation completed in {aggregation_end_time - total_start_time:.2f} seconds.")

    # Data Preprocessing
    logger.info("Starting data preprocessing...")
    preprocessing_start_time = time.time()

    # Merged and Cleaned Data
    cleaned_filename = f"{ticker}_{date_range}_cleaned.csv"
    cleaned_filepath = os.path.join(combined_dir, cleaned_filename)

    if os.path.exists(cleaned_filepath):
        logger.info(f"Cleaned data already exists at {cleaned_filepath}. Loading data...")
        cleaned_df = pd.read_csv(cleaned_filepath)
    else:
        preprocessor = DataPreprocessor(price_df, news_df)
        merged_df = preprocessor.align_data()
        cleaned_df = preprocessor.clean_data(merged_df)
        # Save cleaned_df
        cleaned_df.to_csv(cleaned_filepath, index=False)
        logger.info(f"Cleaned data saved to {cleaned_filepath}")

    preprocessing_end_time = time.time()
    logger.info(f"Data preprocessing completed in {preprocessing_end_time - preprocessing_start_time:.2f} seconds.")

    # Sentiment Analysis
    logger.info("Starting sentiment analysis...")
    sentiment_start_time = time.time()

    sentiment_filename = f"{ticker}_{date_range}_sentiment.csv"
    sentiment_filepath = os.path.join(sentiment_dir, sentiment_filename)

    if os.path.exists(sentiment_filepath):
        logger.info(f"Sentiment data already exists at {sentiment_filepath}. Loading data...")
        cleaned_df = pd.read_csv(sentiment_filepath)
    else:
        sentiment_analyzer = SentimentAnalyzer()
        for column in ['title', 'summary']:
            texts = cleaned_df[column].fillna('').tolist()
            pos_probs, neg_probs, neu_probs, labels = sentiment_analyzer.analyze_sentiment(texts)
            cleaned_df[f'{column}_positive'] = pos_probs
            cleaned_df[f'{column}_negative'] = neg_probs
            cleaned_df[f'{column}_neutral'] = neu_probs
            cleaned_df[f'{column}_sentiment'] = labels
        # Save sentiment-enhanced cleaned_df
        cleaned_df.to_csv(sentiment_filepath, index=False)
        logger.info(f"Sentiment data saved to {sentiment_filepath}")

    sentiment_end_time = time.time()
    logger.info(f"Sentiment analysis completed in {sentiment_end_time - sentiment_start_time:.2f} seconds.")

    # Price Fluctuation Calculation
    logger.info("Calculating price fluctuations...")
    fluctuation_start_time = time.time()

    fluctuation_filename = f"{ticker}_{date_range}_price_fluctuations.csv"
    fluctuation_filepath = os.path.join(pricing_dir, fluctuation_filename)

    if os.path.exists(fluctuation_filepath):
        logger.info(f"Price fluctuation data already exists at {fluctuation_filepath}. Loading data...")
        fluctuation_df = pd.read_csv(fluctuation_filepath)
    else:
        time_periods = {'5_min': 5, '15_min': 15, '30_min': 30, '1_hour': 60}
        fluctuation_calculator = PriceFluctuationCalculator(cleaned_df, time_periods)
        fluctuation_df = fluctuation_calculator.calculate_fluctuations()
        # Save fluctuation_df
        fluctuation_df.to_csv(fluctuation_filepath, index=False)
        logger.info(f"Price fluctuation data saved to {fluctuation_filepath}")

    fluctuation_end_time = time.time()
    logger.info(f"Price fluctuation calculation completed in {fluctuation_end_time - fluctuation_start_time:.2f} seconds.")

    # Technical Indicator Calculation
    logger.info("Calculating technical indicators...")
    indicator_start_time = time.time()

    indicators_filename = f"{ticker}_{date_range}_technical_indicators.csv"
    indicators_filepath = os.path.join(technical_indicators_dir, indicators_filename)

    if os.path.exists(indicators_filepath):
        logger.info(f"Technical indicators data already exists at {indicators_filepath}. Loading data...")
        indicators_df = pd.read_csv(indicators_filepath)
    else:
        indicator_calculator = TechnicalIndicatorCalculator(fluctuation_df)
        indicators_df = indicator_calculator.calculate_rate_of_change(['RSI', 'MACD_Signal'])
        # Save indicators_df
        indicators_df.to_csv(indicators_filepath, index=False)
        logger.info(f"Technical indicators data saved to {indicators_filepath}")

    indicator_end_time = time.time()
    logger.info(f"Technical indicator calculation completed in {indicator_end_time - indicator_start_time:.2f} seconds.")

    # Combine all relevant data into a single DataFrame
    final_filename = f"{ticker}_{date_range}_final.csv"
    final_filepath = os.path.join(combined_dir, final_filename)

    if os.path.exists(final_filepath):
        logger.info(f"Final data already exists at {final_filepath}. Loading data...")
        final_df = pd.read_csv(final_filepath)
    else:
        final_df = indicators_df.copy()
        logger.info(f"Final DataFrame shape: {final_df.shape}")

        # Check if final_df is empty
        if final_df.empty:
            logger.error("Final DataFrame is empty. Cannot proceed further.")
            return

        # Handle missing values in final_df
        logger.info("Handling missing values in final DataFrame...")
        final_df = handle_missing_values(final_df)

        # Save final_df
        final_df.to_csv(final_filepath, index=False)
        logger.info(f"Final data saved to {final_filepath}")

    # Generate string representations of each row
    logger.info("Creating string representations for embedding...")
    embedding_prep_start_time = time.time()

    # Embeddings File
    embeddings_filename = f"{ticker}_{date_range}_embeddings.npy"
    embeddings_filepath = os.path.join(embedded_dir, embeddings_filename)

    if os.path.exists(embeddings_filepath):
        logger.info(f"Embeddings already exist at {embeddings_filepath}. Loading embeddings...")
        embeddings = np.load(embeddings_filepath)
    else:
        def row_to_string(row):
            return ' '.join([f"{key}: {value}" for key, value in row.items() if pd.notnull(value)])

        # Select columns to include in the string representation
        columns_to_include = ["Symbol_x","Open","High","Low","Close","Volume","RSI","MACD","MACD_Signal","MACD_Hist",
                              "title","url","time_published","authors","summary","banner_image","source",
                              "category_within_source","source_domain","topics","overall_sentiment_score",
                              "overall_sentiment_label","ticker_sentiment","Symbol_y","title_positive",
                              "title_negative","title_neutral","title_sentiment","summary_positive",
                              "summary_negative","summary_neutral","summary_sentiment","5_min_change",
                              "5_min_percentage_change","15_min_change","15_min_percentage_change",
                              "30_min_change","30_min_percentage_change","1_hour_change",
                              "1_hour_percentage_change","RSI_roc","MACD_Signal_roc"]

        # Check for missing columns and adjust columns_to_include
        missing_columns = [col for col in columns_to_include if col not in final_df.columns]
        if missing_columns:
            logger.warning(f"The following columns are missing and will be skipped: {missing_columns}")
            columns_to_include = [col for col in columns_to_include if col in final_df.columns]

        # Create the text data for embeddings
        texts_for_embedding = final_df[columns_to_include].apply(row_to_string, axis=1).tolist()
        embedding_prep_end_time = time.time()
        logger.info(f"String representations created in {embedding_prep_end_time - embedding_prep_start_time:.2f} seconds.")

        # Define the instruction for NV-Embed-v2
        instruction = "Instruct: Use the following financial data to predict stock price movements.\nData: "

        # Generate embeddings using DataEmbedder with PCA
        logger.info("Generating embeddings...")
        embedding_start_time = time.time()
        # Set the number of PCA components
        n_pca_components = 512  # Adjust as desired

        # Set the model type ('nvidia' or 'openai')
        model_type = 'openai'  # or 'nvidia'

        # Set the OpenAI model name if using OpenAI
        openai_model_name = 'text-embedding-3-large'  # Adjust as needed

        # Set use_batch_api to True to use the OpenAI Batch API
        use_batch_api = True

        # Initialize the DataEmbedder with PCA
        embedder = DataEmbedder(
            model_type=model_type,
            model_name=openai_model_name,
            n_components=n_pca_components,
            use_batch_api=use_batch_api
        )

        # Generate embeddings (no instruction needed for OpenAI models)
        embeddings = embedder.create_embeddings(texts_for_embedding)

        embedding_end_time = time.time()
        logger.info(f"Embeddings generated in {embedding_end_time - embedding_start_time:.2f} seconds.")

        # Save embeddings
        np.save(embeddings_filepath, embeddings)
        logger.info(f"Embeddings saved to {embeddings_filepath}")

    # Prepare target variable
    target = '5_min_percentage_change'  # Example target
    if target not in final_df.columns:
        logger.error(f"Target column '{target}' not found in DataFrame.")
        return

    y = final_df[target].fillna(0).values

    # Ensure that embeddings and y have the same length
    assert embeddings.shape[0] == len(y), "Mismatch between number of embeddings and target values"

    # Feature Matrix X is the embeddings
    X = embeddings

    # Split data into training, validation, and test sets
    logger.info("Splitting data into training, validation, and test sets...")
    split_start_time = time.time()

    # First, split into training and temporary sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    # Further split the temporary set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    split_end_time = time.time()
    logger.info(f"Data split completed in {split_end_time - split_start_time:.2f} seconds.")

    # Model Training
    logger.info("Training model...")
    training_start_time = time.time()
    input_size = X_train.shape[1]
    hidden_layers = [256, 128, 64]  # Adjusted for reduced-dimensional embeddings
    model = StockPredictor(input_size, hidden_layers)

    # Instantiate ModelTrainer
    trainer = ModelTrainer(model)
    trained_model, training_history = trainer.train(X_train, y_train, X_val, y_val)
    training_end_time = time.time()
    logger.info(f"Model training completed in {training_end_time - training_start_time:.2f} seconds.")

    # Visualize training history using ModelVisualizer
    logger.info("Visualizing training history...")
    visualizer = ModelVisualizer()
    # Log the training history (average loss for each epoch)
    for train_loss, val_loss in zip(training_history['train_loss'], training_history['val_loss']):
        visualizer.log_metrics(train_loss, val_loss)

    # Plot the training and validation loss metrics
    visualizer.plot_metrics()

    # Model Evaluation
    logger.info("Evaluating model...")
    evaluation_start_time = time.time()
    evaluator = ModelEvaluator(trained_model)
    mse, r2 = evaluator.evaluate(X_test, y_test)
    evaluation_end_time = time.time()
    logger.info(f"Model evaluation completed in {evaluation_end_time - evaluation_start_time:.2f} seconds.")

    # Log evaluation results
    logger.info(f"Model evaluation results - MSE: {mse:.4f}, R2 Score: {r2:.4f}")

    # Log total execution time
    total_end_time = time.time()
    logger.info(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")

def handle_missing_values(df):
    """
    Handles missing values in the DataFrame by filling or dropping as appropriate.
    """
    # Convert columns to numeric
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing price data
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].fillna(method='ffill')
    
    # Handle missing technical indicators
    if 'RSI' in df.columns:
        df['RSI'] = df['RSI'].fillna(method='ffill')
    if 'MACD' in df.columns:
        df['MACD'] = df['MACD'].fillna(method='ffill')
    if 'MACD_Signal' in df.columns:
        df['MACD_Signal'] = df['MACD_Signal'].fillna(method='ffill')
    
    # Drop remaining rows with missing price data if any
    df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    
    # Handle missing sentiment scores
    sentiment_columns = ['title_positive', 'title_negative', 'title_neutral',
                         'summary_positive', 'summary_negative', 'summary_neutral']
    for col in sentiment_columns:
        if col in df.columns:
            if 'neutral' in col:
                df[col] = df[col].fillna(1.0)  # Set neutral probability to 1.0
            else:
                df[col] = df[col].fillna(0.0)  # Set positive and negative probabilities to 0.0
    
    return df

if __name__ == '__main__':
    main()