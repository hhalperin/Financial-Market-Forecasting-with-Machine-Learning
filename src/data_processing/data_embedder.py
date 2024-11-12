import os
import time
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import requests
from utils.logger import get_logger
from sklearn.decomposition import PCA

class DataEmbedder:
    """
    Generates embeddings using either NV-Embed-v2 or OpenAI's embedding model.
    Supports synchronous and batch processing.
    """

    def __init__(self, model_type='nvidia', model_name=None, n_components=None, use_batch_api=False):
        """
        Initializes the DataEmbedder.

        Parameters:
            model_type (str): The type of model to use ('nvidia' or 'openai').
            model_name (str): The model name or identifier.
            n_components (int, optional): Number of PCA components for dimensionality reduction.
            use_batch_api (bool): Whether to use the OpenAI Batch API for embedding generation.
        """
        self.model_type = model_type.lower()
        self.use_batch_api = use_batch_api
        self.n_components = n_components
        self.logger = get_logger(self.__class__.__name__)

        # Initialize PCA if n_components is specified
        self.pca = PCA(n_components=n_components) if n_components else None

        if self.model_type == 'nvidia':
            # Import torch and transformers here to avoid unnecessary imports when using OpenAI models
            import torch
            import torch.nn.functional as F
            from transformers import AutoTokenizer, AutoModel

            # Default model name for NVIDIA model if not provided
            self.model_name = model_name or 'nvidia/NV-Embed-v2'
            # Load the NV-Embed-v2 model and tokenizer
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
            self.model.eval()
        elif self.model_type == 'openai':
            # Default model name for OpenAI model if not provided
            self.model_name = model_name or 'text-embedding-ada-002'
            # Ensure that the OpenAI API key is set
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OpenAI API key not found. Set the 'OPENAI_API_KEY' environment variable.")
            self.headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            # Set the OpenAI embedding model
            self.openai_engine = self.model_name
            # Batch API endpoints
            self.files_endpoint = 'https://api.openai.com/v1/files'
            self.batches_endpoint = 'https://api.openai.com/v1/batches'
        else:
            raise ValueError("Invalid model_type. Choose 'nvidia' or 'openai'.")

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

    def submit_batch_job(self, texts):
        """
        Submits a batch job to OpenAI's Batch API for embedding generation.

        Parameters:
            texts (list of str): The texts to embed.

        Returns:
            str: The batch job ID.
        """
        try:
            # Step 1: Prepare the JSONL batch input
            self.logger.info(f"Preparing batch input file for {len(texts)} texts...")
            batch_input = []
            for idx, text in enumerate(texts):
                request = {
                    "custom_id": f"request-{idx}",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": self.openai_engine,
                        "input": text
                    }
                }
                batch_input.append(json.dumps(request))

            batch_input_file_path = "batch_input.jsonl"
            with open(batch_input_file_path, 'w', encoding='utf-8') as f:
                for line in batch_input:
                    f.write(f"{line}\n")

            self.logger.info("Uploading batch input file to OpenAI...")
            with open(batch_input_file_path, 'rb') as f:
                upload_response = requests.post(
                    self.files_endpoint,
                    headers={
                        'Authorization': f'Bearer {self.api_key}'
                    },
                    files={
                        'file': (os.path.basename(batch_input_file_path), f, 'application/jsonl')
                    },
                    data={
                        'purpose': 'batch'
                    }
                )
            upload_response.raise_for_status()
            uploaded_file = upload_response.json()
            batch_input_file_id = uploaded_file['id']
            self.logger.info(f"Batch input file uploaded with ID: {batch_input_file_id}")

            # Step 2: Create the batch job
            self.logger.info("Creating batch job...")
            batch_payload = {
                "input_file_id": batch_input_file_id,
                "completion_window": "24h",
                "endpoint": "/v1/embeddings",
                "metadata": {
                    "description": "Batch embedding generation"
                }
            }
            batch_response = requests.post(
                self.batches_endpoint,
                headers=self.headers,
                json=batch_payload
            )
            batch_response.raise_for_status()
            batch_job = batch_response.json()
            batch_id = batch_job['id']
            self.logger.info(f"Batch job created with ID: {batch_id}")

            # Optionally, delete the batch input file after submission
            os.remove(batch_input_file_path)
            self.logger.info(f"Removed temporary batch input file {batch_input_file_path}")

            return batch_id

        except Exception as e:
            self.logger.error(f"Error submitting batch job: {e}")
            raise

    def retrieve_batch_results(self, batch_id):
        """
        Retrieves the results of a completed batch job.

        Parameters:
            batch_id (str): The ID of the batch job.

        Returns:
            numpy.ndarray: An array containing the embeddings for each text.
        """
        try:
            # Step 3: Poll for batch job completion
            self.logger.info("Waiting for batch job to complete...")
            while True:
                status_response = requests.get(
                    f"{self.batches_endpoint}/{batch_id}",
                    headers=self.headers
                )
                status_response.raise_for_status()
                status_data = status_response.json()
                status = status_data['status']
                self.logger.info(f"Batch status: {status}")
                if status in ["completed", "failed", "cancelled"]:
                    break
                time.sleep(60)  # Increased wait time to 60 seconds to avoid timeout

            if status != "completed":
                self.logger.error(f"Batch job did not complete successfully. Status: {status}")
                raise Exception(f"Batch job failed with status: {status}")

            # Step 4: Retrieve the output file
            output_file_id = status_data.get('output_file_id')
            if not output_file_id:
                self.logger.error("No output_file_id found in batch job response.")
                raise Exception("Batch job completed but no output file ID found.")

            self.logger.info(f"Retrieving output file with ID: {output_file_id}")
            output_file_response = requests.get(
                f"{self.files_endpoint}/{output_file_id}/content",
                headers={
                    'Authorization': f'Bearer {self.api_key}'
                }
            )
            output_file_response.raise_for_status()
            output_content = output_file_response.text

            # Step 5: Process the output file
            self.logger.info("Processing output file...")
            output_embeddings = {}
            for line in output_content.splitlines():
                data = json.loads(line)
                custom_id = data.get('custom_id')
                response_data = data.get('response')
                if response_data and response_data.get('status_code') == 200:
                    embedding = response_data['body']['embedding']
                    idx = int(custom_id.split("-")[1])
                    output_embeddings[idx] = embedding
                else:
                    error = data.get('error')
                    self.logger.error(f"Error in request {custom_id}: {error}")
                    raise Exception(f"Error in batch request {custom_id}: {error}")

            # Ensure the embeddings are in the correct order
            embeddings = [output_embeddings[idx] for idx in sorted(output_embeddings.keys())]
            embeddings = np.array(embeddings)

            # Step 6: Apply PCA if specified
            if self.pca:
                self.logger.info(f"Applying PCA to reduce embeddings to {self.n_components} dimensions.")
                embeddings = self.pca.fit_transform(embeddings)

            self.logger.info(f"Retrieved embeddings for {len(embeddings)} texts from batch job.")
            return embeddings

        except Exception as e:
            self.logger.error(f"Error retrieving batch results: {e}")
            raise
