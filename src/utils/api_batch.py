# File: api_batch.py

import os
import json
import numpy as np
import pandas as pd
from data_processing.data_embedder import DataEmbedder
from logger import get_logger

logger = get_logger('APIBatch')

def submit_batch_job(ticker, start_date, end_date, final_df):
    """
    Submits a batch job to OpenAI's Batch API for embedding generation.
    """
    # Define paths
    embedded_dir = os.path.join('data', 'embedded')
    os.makedirs(embedded_dir, exist_ok=True)
    batch_id_filepath = os.path.join(embedded_dir, 'batch_id.json')
    embeddings_filepath = os.path.join(embedded_dir, 'embeddings.npy')

    # Check if embeddings already exist
    if os.path.exists(embeddings_filepath):
        logger.info(f"Embeddings file {embeddings_filepath} already exists. No need to submit a new batch job.")
        return None

    # Check if a batch job is already submitted
    if os.path.exists(batch_id_filepath):
        logger.info(f"Batch ID file {batch_id_filepath} exists. A batch job might be in progress.")
        return None

    # Prepare texts for embedding
    def row_to_string(row):
        return ' '.join([f"{key}: {value}" for key, value in row.items() if pd.notnull(value)])

    columns_to_include = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal',
        'title_positive', 'title_negative', 'title_neutral',
        'summary_positive', 'summary_negative', 'summary_neutral'
    ]

    # Check if the required columns exist in the DataFrame
    missing_columns = [col for col in columns_to_include if col not in final_df.columns]
    if missing_columns:
        logger.warning(f"The following columns are missing and will be skipped: {missing_columns}")
        columns_to_include = [col for col in columns_to_include if col in final_df.columns]

    texts_for_embedding = final_df[columns_to_include].apply(row_to_string, axis=1).tolist()

    # Initialize DataEmbedder
    embedder = DataEmbedder(
        model_type='openai',
        model_name='text-embedding-ada-002',
        n_components=512,
        use_batch_api=True
    )

    # Submit batch job
    logger.info("Submitting batch job...")
    batch_id = embedder.submit_batch_job(texts_for_embedding)
    logger.info(f"Batch job submitted with ID: {batch_id}")

    # Save batch_id to a file
    with open(batch_id_filepath, 'w') as f:
        json.dump({'batch_id': batch_id}, f)
    logger.info(f"Batch ID saved to {batch_id_filepath}")

    return batch_id

def retrieve_batch_results(batch_id):
    """
    Retrieves the results of a completed batch job and saves embeddings to file.
    """
    # Define paths
    embedded_dir = os.path.join('data', 'embedded')
    os.makedirs(embedded_dir, exist_ok=True)
    embeddings_filepath = os.path.join(embedded_dir, 'embeddings.npy')

    # Check if embeddings already exist
    if os.path.exists(embeddings_filepath):
        logger.info(f"Embeddings file {embeddings_filepath} already exists. No need to retrieve embeddings.")
        return np.load(embeddings_filepath)

    if not batch_id:
        logger.error("No batch ID provided.")
        return None

    # Initialize DataEmbedder
    embedder = DataEmbedder(
        model_type='openai',
        model_name='text-embedding-ada-002',
        n_components=512,
        use_batch_api=True
    )

    # Retrieve batch results
    logger.info(f"Retrieving embeddings for batch ID: {batch_id}")
    embeddings = embedder.retrieve_batch_results(batch_id)

    # Save embeddings
    np.save(embeddings_filepath, embeddings)
    logger.info(f"Embeddings saved to {embeddings_filepath}")

    return embeddings
