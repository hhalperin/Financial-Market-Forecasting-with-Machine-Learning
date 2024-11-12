# File: retrieve_batch.py

import os
import json
import numpy as np
import argparse
from data_processing.data_embedder import DataEmbedder
from utils.logger import get_logger

def main(batch_id=None):
    logger = get_logger('RetrieveBatch')
    # Define paths
    embedded_dir = os.path.join('data', 'embedded')
    os.makedirs(embedded_dir, exist_ok=True)
    embeddings_filepath = os.path.join(embedded_dir, 'embeddings.npy')

    # Check if embeddings already exist
    if os.path.exists(embeddings_filepath):
        logger.info(f"Embeddings file {embeddings_filepath} already exists. No need to retrieve embeddings.")
        return

    # If batch_id is not provided, prompt the user
    if not batch_id:
        batch_id = input("Enter the batch ID: ").strip()
        if not batch_id:
            logger.error("No batch ID provided.")
            return

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve embeddings from a batch job.')
    parser.add_argument('--batch_id', type=str, help='The batch job ID to retrieve embeddings for.')
    args = parser.parse_args()
    main(batch_id=args.batch_id)

# batch_67330e8849088190af08c593e35c33d0