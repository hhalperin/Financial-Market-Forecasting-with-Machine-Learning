"""
Data Embedder Module

Generates text embeddings from input texts using a Hugging Face transformer model.
Optionally applies PCA for dimensionality reduction.
"""

import warnings
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing import List
import pandas as pd
from src.utils.logger import get_logger

# Suppress specific warnings from Hugging Face libraries for cleaner output.
warnings.filterwarnings("ignore", message="resume_download is deprecated and will be removed in version 1.0.0")
warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False")

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
        Initializes the DataEmbedder for generating text embeddings using a Hugging Face model.

        Args:
            model_name (str): Hugging Face model name (default: "gme-qwen2-vl2b").
            n_components (int): Number of PCA components for dimensionality reduction (default: 128).
            batch_size (int): Batch size for processing texts (default: 8).
            use_pca (bool): Whether to apply PCA after generating embeddings (default: True).
            combine_fields (bool): If True, combine multiple text fields into a single composite text.
            fields_to_combine (List[str]): List of fields to combine (if combine_fields is True).
            combine_template (str): Template to combine fields.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.model_name = model_name
        self.n_components = n_components
        self.batch_size = batch_size
        self.use_pca = use_pca
        self.combine_fields = combine_fields
        self.fields_to_combine = fields_to_combine if fields_to_combine is not None else []
        self.combine_template = combine_template

        # Set device to GPU if available, otherwise CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

        # Initialize PCA if enabled.
        self.pca = PCA(n_components=self.n_components) if (self.use_pca and self.n_components > 0) else None

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts using batch processing and optional PCA.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            np.ndarray: Array of embeddings; if PCA is enabled, reduced to n_components dimensions.
        """
        self.logger.info(f"Generating embeddings for {len(texts)} texts...")
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding Batches"):
                batch_texts = texts[i:i + self.batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                outputs = self.model(**inputs)
                # Compute mean token embedding and L2 normalize.
                embeddings = F.normalize(outputs.last_hidden_state.mean(dim=1), p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        embeddings = np.vstack(all_embeddings)

        # Apply PCA for dimensionality reduction if enabled.
        if self.pca:
            self.logger.info(f"Applying PCA to reduce dimensions to {self.n_components}.")
            embeddings = self.pca.fit_transform(embeddings)

        self.logger.info("Embedding generation completed.")
        return embeddings

    def embed_columns(self, df: pd.DataFrame, columns_to_embed: List[str]) -> pd.DataFrame:
        """
        Generates embeddings for specified DataFrame columns. If composite embedding is enabled,
        combines specified fields into one column first.

        Args:
            df (pd.DataFrame): DataFrame with text columns.
            columns_to_embed (List[str]): Column names to generate embeddings for (if not combining).

        Returns:
            pd.DataFrame: DataFrame with original text columns replaced by embedding columns.
        """
        df_embedded = df.copy()

        if self.combine_fields and self.fields_to_combine and all(col in df.columns for col in self.fields_to_combine):
            # Create composite text using a helper function.
            def combine_fields_func(row):
                return self.combine_template.format(**{field: str(row.get(field, "")) for field in self.fields_to_combine})
            df_embedded["composite_text"] = df_embedded.apply(combine_fields_func, axis=1)
            texts = df_embedded["composite_text"].tolist()
            embeddings = self.generate_embeddings(texts)
            embedding_df = pd.DataFrame(
                embeddings,
                columns=[f"composite_embedding_{i}" for i in range(embeddings.shape[1])],
                index=df_embedded.index
            )
            # Drop original fields and composite column; then concatenate embeddings.
            df_embedded = pd.concat(
                [df_embedded.drop(columns=self.fields_to_combine + ["composite_text"]), embedding_df],
                axis=1
            )
            self.logger.info("Generated composite embeddings.")
        else:
            for col in columns_to_embed:
                if col not in df_embedded.columns:
                    self.logger.warning(f"Column '{col}' not found; skipping embedding.")
                    continue
                texts = df_embedded[col].fillna("").astype(str).tolist()
                embeddings = self.generate_embeddings(texts)
                embedding_df = pd.DataFrame(
                    embeddings,
                    columns=[f"{col}_embedding_{i}" for i in range(embeddings.shape[1])],
                    index=df_embedded.index
                )
                # Replace original column with its embeddings.
                df_embedded = pd.concat([df_embedded.drop(columns=[col]), embedding_df], axis=1)
                self.logger.info(f"Generated embeddings for column '{col}'.")
        return df_embedded

    def save_embeddings(self, embeddings: np.ndarray, filename: str, data_handler, stage: str = "embeddings") -> None:
        """
        Saves the generated embeddings via the provided DataHandler.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            filename (str): File name for saving.
            data_handler: DataHandler instance.
            stage (str): Storage stage (default: "embeddings").
        """
        data_handler.save_data(embeddings, filename, data_type="embeddings", stage=stage)
        self.logger.info(f"Embeddings saved to {stage}/{filename}.")
