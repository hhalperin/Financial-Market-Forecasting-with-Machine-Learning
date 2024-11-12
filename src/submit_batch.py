# File: submit_batch.py

import os
import json
import pandas as pd
from data_processing.data_embedder import DataEmbedder
from utils.logger import get_logger

def main():
    logger = get_logger('SubmitBatch')
    # Define paths
    embedded_dir = os.path.join('data', 'embedded')
    os.makedirs(embedded_dir, exist_ok=True)
    batch_id_filepath = os.path.join(embedded_dir, 'batch_id.json')

    # Check if embeddings already exist
    embeddings_filepath = os.path.join(embedded_dir, 'embeddings.npy')
    if os.path.exists(embeddings_filepath):
        logger.info(f"Embeddings file {embeddings_filepath} already exists. No need to submit a new batch job.")
        return

    # Check if a batch job is already submitted
    if os.path.exists(batch_id_filepath):
        logger.info(f"Batch ID file {batch_id_filepath} exists. A batch job might be in progress.")
        return

    # Load your final_df from the combined directory
    final_filepath = os.path.join('data', 'combined', 'AAPL_2023-01-01_to_2023-01-31_final.csv')
    if not os.path.exists(final_filepath):
        logger.error(f"Final data file {final_filepath} not found. Please run the data preprocessing steps first.")
        return

    final_df = pd.read_csv(final_filepath)

    # Prepare texts for embedding
    def row_to_string(row):
        return ' '.join([f"{key}: {value}" for key, value in row.items() if pd.notnull(value)])

    columns_to_include = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal',
                          'title_positive', 'title_negative', 'title_neutral',
                          'summary_positive', 'summary_negative', 'summary_neutral']

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

if __name__ == '__main__':
    main()
