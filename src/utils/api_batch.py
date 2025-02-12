import os
import json
import numpy as np
import pandas as pd
from src.data_processing.data_embedder import DataEmbedder
from src.utils.logger import get_logger

logger = get_logger('APIBatch')

def submit_batch_job(ticker: str, start_date: str, end_date: str, final_df: pd.DataFrame) -> str:
    """
    Submits a batch job for generating embeddings via OpenAI's API.
    
    :param ticker: Stock ticker.
    :param start_date: Start date of data.
    :param end_date: End date of data.
    :param final_df: DataFrame with data for which embeddings are needed.
    :return: The batch job ID as a string.
    """
    embedded_dir = os.path.join('data', 'embedded')
    os.makedirs(embedded_dir, exist_ok=True)
    batch_id_filepath = os.path.join(embedded_dir, 'batch_id.json')
    embeddings_filepath = os.path.join(embedded_dir, 'embeddings.npy')
    if os.path.exists(embeddings_filepath):
        logger.info(f"Embeddings file {embeddings_filepath} already exists. Skipping batch submission.")
        return ""
    if os.path.exists(batch_id_filepath):
        logger.info(f"Batch ID file {batch_id_filepath} exists. A batch job might be in progress.")
        with open(batch_id_filepath, 'r') as f:
            data = json.load(f)
            return data.get('batch_id', '')
    def row_to_string(row: pd.Series) -> str:
        return ' '.join([f"{key}: {value}" for key, value in row.items() if pd.notnull(value)])
    columns_to_include = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal',
        'title_positive', 'title_negative', 'title_neutral',
        'summary_positive', 'summary_negative', 'summary_neutral'
    ]
    missing_columns = [col for col in columns_to_include if col not in final_df.columns]
    if missing_columns:
        logger.warning(f"Missing columns: {missing_columns}. They will be skipped.")
        columns_to_include = [col for col in columns_to_include if col in final_df.columns]
    texts_for_embedding = final_df[columns_to_include].apply(row_to_string, axis=1).tolist()
    # Initialize DataEmbedder with OpenAI settings.
    embedder = DataEmbedder(
        model_type='openai',
        model_name='text-embedding-ada-002',
        n_components=512,
        use_batch_api=True
    )
    logger.info("Submitting batch job for embeddings...")
    batch_id = embedder.submit_batch_job(texts_for_embedding)
    logger.info(f"Batch job submitted with ID: {batch_id}")
    with open(batch_id_filepath, 'w') as f:
        json.dump({'batch_id': batch_id}, f)
    return batch_id

def retrieve_batch_results(batch_id: str) -> np.ndarray:
    """
    Retrieves embeddings from a completed batch job.
    
    :param batch_id: The batch job ID.
    :return: NumPy array of embeddings.
    """
    embedded_dir = os.path.join('data', 'embedded')
    os.makedirs(embedded_dir, exist_ok=True)
    embeddings_filepath = os.path.join(embedded_dir, 'embeddings.npy')
    if os.path.exists(embeddings_filepath):
        logger.info(f"Embeddings file {embeddings_filepath} exists. Loading embeddings.")
        return np.load(embeddings_filepath)
    if not batch_id:
        logger.error("No batch ID provided.")
        return np.array([])
    embedder = DataEmbedder(
        model_type='openai',
        model_name='text-embedding-ada-002',
        n_components=512,
        use_batch_api=True
    )
    logger.info(f"Retrieving embeddings for batch ID: {batch_id}")
    embeddings = embedder.retrieve_batch_results(batch_id)
    np.save(embeddings_filepath, embeddings)
    logger.info(f"Embeddings saved to {embeddings_filepath}")
    return embeddings
