"""
Data Embedder Module

Handles embedding generation for text data using Hugging Face models.
Supports batching and optional dimensionality reduction (PCA). Unlike before, when embedding
a text column, the entire embedding vector is stored in a single new column (e.g., 'title_embedding'),
replacing the original text column.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing import List
import pandas as pd
from src.utils.logger import get_logger

class DataEmbedder:
    """
    Generates embeddings for text data. Embeddings for a text column replace the original column
    with a new column (named '<original>_embedding') that holds the full embedding vector.
    
    Configurable parameters (preferably set via config.py) include:
      - model_name: Hugging Face model identifier.
      - n_components: Dimensionality after PCA reduction.
      - batch_size: Number of texts per batch.
      - use_pca: Flag to apply PCA.
    """
    def __init__(self, 
                 model_name: str = "gme-qwen2-vl2b", 
                 n_components: int = 128, 
                 batch_size: int = 8, 
                 use_pca: bool = True) -> None:
        """
        Initializes the DataEmbedder.

        :param model_name: Hugging Face model name.
        :param n_components: Number of dimensions to keep after PCA (if enabled).
        :param batch_size: Batch size for embedding generation.
        :param use_pca: Whether to apply PCA for dimensionality reduction.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.model_name = model_name
        self.n_components = n_components
        self.batch_size = batch_size
        self.use_pca = use_pca

        # Initialize the tokenizer and model from Hugging Face.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self.pca = PCA(n_components=self.n_components) if self.use_pca and self.n_components > 0 else None

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        :param texts: List of strings to embed.
        :return: NumPy array of embeddings.
        """
        self.logger.info(f"Generating embeddings for {len(texts)} texts...")
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding Batches"):
                batch_texts = texts[i:i + self.batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
                outputs = self.model(**inputs)
                # Mean pooling with L2 normalization.
                embeddings = F.normalize(outputs.last_hidden_state.mean(dim=1), p=2, dim=1).cpu().numpy()
                all_embeddings.append(embeddings)
        embeddings = np.vstack(all_embeddings)
        if self.pca:
            self.logger.info(f"Applying PCA to reduce dimensions to {self.n_components}.")
            embeddings = self.pca.fit_transform(embeddings)
        self.logger.info("Embedding generation completed.")
        return embeddings

    def embed_columns(self, df: pd.DataFrame, columns_to_embed: List[str]) -> pd.DataFrame:
        """
        Replaces each specified text column in the DataFrame with a new column that contains the
        entire embedding vector. The new column is named '<original_column>_embedding'.

        :param df: Input DataFrame.
        :param columns_to_embed: List of column names to embed.
        :return: DataFrame with embedded columns.
        """
        df_embedded = df.copy()
        for col in columns_to_embed:
            if col not in df_embedded.columns:
                self.logger.warning(f"Column '{col}' not found in DataFrame; skipping embedding.")
                continue
            texts = df_embedded[col].fillna('').astype(str).tolist()
            self.logger.info(f"Embedding column '{col}' with {len(texts)} texts.")
            embeddings = self.generate_embeddings(texts)
            # Save the embedding vector as a list in a single new column.
            # We convert each row's embedding vector to a list.
            df_embedded[f"{col}_embedding"] = [emb.tolist() for emb in embeddings]
            # Drop the original text column.
            df_embedded.drop(columns=[col], inplace=True)
            self.logger.info(f"Replaced column '{col}' with '{col}_embedding'.")
        return df_embedded

    def save_embeddings(self, embeddings: np.ndarray, filename: str, data_handler, stage: str = 'embeddings') -> None:
        """
        Saves embeddings to a file using the provided data handler.

        :param embeddings: NumPy array of embeddings.
        :param filename: File name for saving.
        :param data_handler: DataHandler instance to perform the save.
        :param stage: Stage or folder name (e.g., 'embeddings').
        """
        data_handler.save_data(embeddings, filename, data_type='embeddings', stage=stage)
        self.logger.info(f"Embeddings saved to {stage}/{filename}.")
