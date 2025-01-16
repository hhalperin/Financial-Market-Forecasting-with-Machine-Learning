# test_all.py

import os
import time
from dotenv import load_dotenv
from utils.logger import get_logger

# Import the main() functions from each module.
from data_aggregate_main import main as aggregator_main
from data_processing_main import main as processing_main
from embedding_main import main as embedding_main
from ml_training_main import main as ml_training_main

logger = get_logger("TestAllPipeline")

def main():
    load_dotenv()
    start_time = time.time()

    logger.info("--- Starting Data Aggregation ---")
    aggregator_main()

    logger.info("--- Starting Data Processing ---")
    processing_main()

    logger.info("--- Starting Embedding Generation ---")
    embedding_main()

    logger.info("--- Starting ML Training ---")
    ml_training_main()

    elapsed = time.time() - start_time
    logger.info(f"=== Entire pipeline completed in {elapsed:.2f}s ===")

if __name__ == "__main__":
    main()
