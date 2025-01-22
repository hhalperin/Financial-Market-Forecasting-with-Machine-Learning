import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from tqdm import tqdm
from src.utils.logger import get_logger


class DataEmbedder:
    """
    Handles embedding generation for text data using Hugging Face models.
    Supports batching and optional dimensionality reduction (PCA).
    """

    def __init__(self, 
                 model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 n_components=128, 
                 batch_size=8, 
                 use_pca=True):
        """
        :param model_name: Hugging Face model name.
        :param n_components: Number of dimensions to keep after PCA (if enabled).
        :param batch_size: Number of texts per batch.
        :param use_pca: Whether to apply PCA for dimensionality reduction.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.model_name = model_name
        self.n_components = n_components
        self.batch_size = batch_size
        self.use_pca = use_pca

        # Initialize model and PCA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self.pca = PCA(n_components=self.n_components) if self.use_pca and self.n_components > 0 else None

    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts using the model.
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
                embeddings = F.normalize(outputs.last_hidden_state.mean(dim=1), p=2, dim=1).cpu().numpy()
                all_embeddings.append(embeddings)

        embeddings = np.vstack(all_embeddings)
        if self.pca:
            self.logger.info(f"Applying PCA to reduce dimensions to {self.n_components}.")
            embeddings = self.pca.fit_transform(embeddings)

        self.logger.info("Embedding generation completed.")
        return embeddings

    def create_embeddings_from_dataframe(self, df, columns_to_embed=None):
        """
        Generate embeddings from a DataFrame's specified columns or entire rows.
        :param df: Input DataFrame.
        :param columns_to_embed: Columns to combine into text for embedding.
        :return: DataFrame with an 'embedding' column.
        """
        if columns_to_embed:
            texts = df[columns_to_embed].fillna('').astype(str).agg(' '.join, axis=1).tolist()
        else:
            texts = df.apply(lambda row: ' '.join(f"{k}: {v}" for k, v in row.items() if pd.notnull(v)), axis=1).tolist()

        embeddings = self.generate_embeddings(texts)
        df['embedding'] = list(embeddings)
        self.logger.info(f"Embeddings added to DataFrame. Shape: {df.shape}")
        return df

    def save_embeddings(self, embeddings, filename, data_handler, stage='embeddings'):
        """
        Save embeddings to a file using the provided data handler.
        :param embeddings: NumPy array of embeddings.
        :param filename: Name of the file to save.
        :param data_handler: Data handler instance.
        :param stage: Storage stage (e.g., 'embeddings').
        """
        data_handler.save_data(embeddings, filename, data_type='embeddings', stage=stage)
        self.logger.info(f"Embeddings saved to {stage}/{filename}.")
