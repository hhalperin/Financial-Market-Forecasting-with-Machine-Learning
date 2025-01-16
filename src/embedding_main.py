# embed_only.py

import os
import time
from dotenv import load_dotenv
import pandas as pd

from utils.logger import get_logger
from utils.data_handler import DataHandler
from data_processing.data_embedder import DataEmbedder

logger = get_logger("EmbedOnly")

def main():
    start_time = time.time()
    load_dotenv()  # Load .env if needed

    local_mode_str = os.getenv("LOCAL_MODE", "true").lower()
    local_mode = (local_mode_str == 'true')
    storage_mode = 'local' if local_mode else 's3'

    logger.info(f"[embed_only] local_mode={local_mode}, storage_mode={storage_mode}")

    # 1) Build data handler
    if storage_mode == 'local':
        data_handler = DataHandler(base_data_dir='data', storage_mode='local')
    else:
        s3_bucket = os.getenv("S3_BUCKET", None)
        if not s3_bucket:
            raise ValueError("S3_BUCKET not set for cloud mode.")
        data_handler = DataHandler(bucket=s3_bucket, base_data_dir='data', storage_mode='s3')

    # 2) Load the preprocessed DataFrame from stage='preprocessed'
    ticker = 'AAPL'
    date_range = '2023-01-01_to_2024-01-31'

    def no_op_fetch():
        # If the CSV doesn't exist, do nothing. We expect it to be there
        return pd.DataFrame()

    preprocessed_df = data_handler(
        ticker, date_range, 'preprocessed', no_op_fetch, stage='preprocessed'
    )
    if preprocessed_df is None or preprocessed_df.empty:
        logger.error("No preprocessed data found! Make sure it exists in data/preprocessed/.")
        return

    logger.info(f"[embed_only] Loaded preprocessed data shape={preprocessed_df.shape}")

    # 3) Set up the embedder
    embedder = DataEmbedder(
        model_type='nvidia',  # or if you have an openai approach
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        n_components=128,
        local_mode=True,      # if you want local huggingface
        use_pca=True
    )

    # 4) Actually create embeddings from selected columns (e.g., 'title','summary')
    def fetch_embeddings():
        return embedder.create_embeddings_from_dataframe(
            preprocessed_df,
            ticker,
            date_range,
            data_handler,
            columns_to_embed=['title','summary']
        )

    # Let DataHandler store them in stage='embeddings'
    embeddings = data_handler(
        ticker, 
        date_range, 
        'embeddings', 
        fetch_embeddings, 
        stage='embeddings'
    )
    if embeddings is not None:
        logger.info(f"[embed_only] embeddings shape={embeddings.shape}")
    else:
        logger.warning("[embed_only] No embeddings created or fetched.")

    elapsed = time.time() - start_time
    logger.info(f"[embed_only] Done. Time elapsed={elapsed:.2f}s")

if __name__ == "__main__":
    main()
