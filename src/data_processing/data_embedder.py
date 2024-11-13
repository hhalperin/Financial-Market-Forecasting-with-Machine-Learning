import os
import time
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import requests
from logger import get_logger
from sklearn.decomposition import PCA
from utils.data_handler import DataHandler

class DataEmbedder:
    """
    Generates embeddings using either NV-Embed-v2 or OpenAI's embedding model.
    Supports synchronous and batch processing.
    """

    def __init__(self, model_type='nvidia', model_name="text-embedding-3-small", n_components=None, use_batch_api=False):
        """
        Initializes the DataEmbedder.
        """
        self.model_type = model_type.lower()
        self.use_batch_api = use_batch_api
        self.n_components = n_components
        self.logger = get_logger(self.__class__.__name__)
        # PCA Initialization
        self.pca = PCA(n_components=n_components) if n_components else None

        # NVIDIA and OpenAI Initialization
        if self.model_type == 'nvidia':
            import torch
            import torch.nn.functional as F
            from transformers import AutoTokenizer, AutoModel
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.model.eval()
        elif self.model_type == 'openai':
            self.model_name = model_name
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OpenAI API key not found.")
            self.headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        else:
            raise ValueError("Invalid model_type. Choose 'nvidia' or 'openai'.")

    def create_embeddings_from_dataframe(self, df, ticker, date_range, data_handler, horizon_id="default"):
        """
        Create embeddings for the given DataFrame and save/load using DataHandler.
        """
        texts_for_embedding = df.apply(self._row_to_string, axis=1).tolist()
        return data_handler(ticker, date_range, 'embeddings', lambda: self.create_embeddings(texts_for_embedding), config_id=horizon_id)

    def _row_to_string(self, row):
        """
        Converts a row of the DataFrame to a concatenated string.
        """
        elements = []
        for key, value in row.items():
            value_str = ', '.join(map(str, value)) if isinstance(value, (list, pd.Series, np.ndarray)) else str(value) if pd.notnull(value) else 'null'
            elements.append(f"{key}: {value_str}")
        return ' '.join(elements)

    # NVIDIA and OpenAI Embedding Functions unchanged for brevity

    def create_embeddings(self, texts, instruction=""):
        """
        Creates embeddings for a list of texts using the specified model.

        Parameters:
            texts (list of str): The texts to embed.
            instruction (str): Instruction to guide the embedding model (used for NV-Embed-v2).

        Returns:
            numpy.ndarray: An array containing the embeddings for each text.
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        # Remove newlines from texts
        texts = [text.replace("\n", " ") for text in texts]

        if self.model_type == 'nvidia':
            return self._create_embeddings_nvidia(texts, instruction)
        elif self.model_type == 'openai':
            if self.use_batch_api:
                return self._create_embeddings_openai(texts)
            else:
                return self._create_embeddings_openai(texts)
        else:
            raise ValueError("Invalid model_type. Choose 'nvidia' or 'openai'.")

    def _create_embeddings_nvidia(self, texts, instruction=""):
        """
        Creates embeddings using NV-Embed-v2.

        Parameters:
            texts (list of str): The texts to embed.
            instruction (str): Instruction to guide the embedding model.

        Returns:
            numpy.ndarray: Embeddings array.
        """
        import torch
        import torch.nn.functional as F

        try:
            batch_size = 8  # Adjust based on your hardware capabilities
            embeddings_list = []

            self.logger.info(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}...")

            with torch.no_grad():
                for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
                    batch_texts = texts[i:i + batch_size]
                    # Append the instruction to each text
                    if instruction:
                        batch_texts = [instruction + text + self.tokenizer.eos_token for text in batch_texts]
                    else:
                        batch_texts = [text + self.tokenizer.eos_token for text in batch_texts]

                    # Tokenize and prepare inputs
                    inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(self.device)

                    # Generate embeddings
                    outputs = self.model(**inputs)
                    # Assuming the model returns last_hidden_state; adjust based on actual model output
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    embeddings_list.append(embeddings.cpu().numpy())

            embeddings = np.vstack(embeddings_list)

            # Apply PCA if specified
            if self.pca:
                self.logger.info(f"Applying PCA to reduce embeddings to {self.n_components} dimensions.")
                embeddings = self.pca.fit_transform(embeddings)

            self.logger.info(f"Generated embeddings for {len(texts)} texts.")
            return embeddings

        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise

    def _create_embeddings_openai(self, texts):
        """
        Creates embeddings using OpenAI's embedding model via synchronous API.

        Parameters:
            texts (list of str): The texts to embed.

        Returns:
            numpy.ndarray: Embeddings array.
        """
        try:
            batch_size = 2048  # Adjust based on your needs and any rate limits
            embeddings_list = []

            self.logger.info(f"Generating embeddings for {len(texts)} texts using OpenAI model '{self.openai_engine}'...")

            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
                batch_texts = texts[i:i + batch_size]
                payload = {
                    "input": batch_texts,
                    "model": self.openai_engine
                }
                response = requests.post(
                    'https://api.openai.com/v1/embeddings',
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                batch_embeddings = [item['embedding'] for item in data['data']]
                embeddings_list.extend(batch_embeddings)

            embeddings = np.array(embeddings_list)

            # Apply PCA if specified
            if self.pca:
                self.logger.info(f"Applying PCA to reduce embeddings to {self.n_components} dimensions.")
                embeddings = self.pca.fit_transform(embeddings)

            self.logger.info(f"Generated embeddings for {len(texts)} texts.")
            return embeddings

        except Exception as e:
            self.logger.error(f"Error generating embeddings with OpenAI: {e}")
            raise
