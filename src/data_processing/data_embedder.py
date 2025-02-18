"""
Data Embedder Module

Handles embedding generation for text data using Hugging Face models.
Supports batching and optional dimensionality reduction (PCA). If composite embedding is enabled,
multiple text fields (e.g., authors, title, summary) are concatenated into a single string
using a configurable template, and the resulting embedding replaces the original columns.

Note:
- We suppress warnings from huggingface_hub and transformers regarding `resume_download`
  and `weights_only` to keep the output clean.
"""

import warnings

# Suppress the resume_download FutureWarning from huggingface_hub.
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0")
# Suppress the torch.load warning regarding weights_only.
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

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
    def __init__(self, 
                 model_name: str = "gme-qwen2-vl2b", 
                 n_components: int = 128, 
                 batch_size: int = 8, 
                 use_pca: bool = True,
                 combine_fields: bool = False,
                 fields_to_combine: List[str] = None,
                 combine_template: str = "authors: {authors}; title: {title}; summary: {summary}"
                ) -> None:
        """
        Initializes the DataEmbedder.

        :param model_name: Hugging Face model name for embeddings.
        :param n_components: Number of PCA components (if PCA is used).
        :param batch_size: Batch size for processing texts.
        :param use_pca: Whether to apply PCA for dimensionality reduction.
        :param combine_fields: If True, combine multiple text fields into one composite text.
        :param fields_to_combine: List of field names to combine (used if combine_fields is True).
        :param combine_template: Template to format combined text. Fields in curly braces are replaced.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.model_name = model_name
        self.n_components = n_components
        self.batch_size = batch_size
        self.use_pca = use_pca
        self.combine_fields = combine_fields
        self.fields_to_combine = fields_to_combine if fields_to_combine is not None else []
        self.combine_template = combine_template

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Load the model normally, as weights_only is not supported by this model.
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self.pca = PCA(n_components=self.n_components) if self.use_pca and self.n_components > 0 else None

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts using batching.
        
        :param texts: List of input texts.
        :return: NumPy array of embeddings.
        """
        self.logger.info(f"Generating embeddings for {len(texts)} texts...")
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding Batches"):
                batch_texts = texts[i:i + self.batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
                outputs = self.model(**inputs)
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
        Generates embeddings for specified columns.
        If composite embedding is enabled, combines the specified fields into one text per row.

        :param df: Input DataFrame.
        :param columns_to_embed: List of column names to embed.
        :return: DataFrame with new embedding column(s) replacing the original text columns.
        """
        df_embedded = df.copy()
        if self.combine_fields and self.fields_to_combine:
            # Create a composite text column using the combine_template.
            def combine_fields_func(row):
                # Prepare a dictionary with keys from fields_to_combine; use empty string if missing.
                field_values = {field: str(row.get(field, "")) for field in self.fields_to_combine}
                return self.combine_template.format(**field_values)
            df_embedded["composite_text"] = df_embedded.apply(combine_fields_func, axis=1)
            self.logger.info(f"Generating composite embeddings for {len(df_embedded)} rows.")
            texts = df_embedded["composite_text"].tolist()
            embeddings = self.generate_embeddings(texts)
            df_embedded["composite_embedding"] = [emb.tolist() for emb in embeddings]
            # Drop the original fields used for composite embedding.
            df_embedded.drop(columns=self.fields_to_combine + ["composite_text"], inplace=True)
            self.logger.info("Replaced original fields with 'composite_embedding'.")
        else:
            # Process each specified column separately.
            for col in columns_to_embed:
                if col not in df_embedded.columns:
                    self.logger.warning(f"Column '{col}' not found in DataFrame; skipping embedding.")
                    continue
                texts = df_embedded[col].fillna('').astype(str).tolist()
                self.logger.info(f"Embedding column '{col}' with {len(texts)} texts.")
                embeddings = self.generate_embeddings(texts)
                df_embedded[f"{col}_embedding"] = [emb.tolist() for emb in embeddings]
                df_embedded.drop(columns=[col], inplace=True)
                self.logger.info(f"Replaced column '{col}' with '{col}_embedding'.")
        return df_embedded

    def save_embeddings(self, embeddings: np.ndarray, filename: str, data_handler, stage: str = 'embeddings') -> None:
        """
        Saves embeddings using the provided data_handler.

        :param embeddings: NumPy array of embeddings.
        :param filename: Filename to save the embeddings under.
        :param data_handler: Instance of DataHandler to perform the save.
        :param stage: Storage stage/category.
        """
        data_handler.save_data(embeddings, filename, data_type='embeddings', stage=stage)
        self.logger.info(f"Embeddings saved to {stage}/{filename}.")
