"""Ollama embedding utilities for RAG."""
import os
from typing import List
import numpy as np
import requests


class OllamaEmbedder:
    """Wrapper for Ollama embedding model (local)."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        """
        Initialize Ollama embedder.

        Args:
            base_url: Ollama server base URL (default: http://localhost:11434)
            model: Embedding model name (e.g., 'nomic-embed-text' or 'all-minilm')
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text using Ollama embeddings API."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding")
            if embedding is None:
                raise RuntimeError("No 'embedding' field in Ollama response")
            return embedding
        except Exception as e:
            raise Exception(f"Error generating embedding via Ollama: {str(e)}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed_text(t) for t in texts]

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query (for search)."""
        emb = self.embed_text(query)
        return np.array(emb, dtype=np.float32)

"""Ollama-based embedding utilities for RAG."""
import os
import requests
import numpy as np
from typing import List

class OllamaEmbedder:
    """Wrapper for Ollama embedding model."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        """
        Initialize Ollama embedder.
        
        Args:
            base_url: Ollama API base URL
            model: Model name to use (e.g., 'llama2', 'mistral', etc.)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        
        # Do not hard-fail on availability checks during app import
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using Ollama.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get('embedding', [])
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text))
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query (for search).
        
        Args:
            query: Query text
            
        Returns:
            Embedding as numpy array
        """
        embedding = self.embed_text(query)
        return np.array(embedding, dtype=np.float32)