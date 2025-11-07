"""Gemini embedding utilities for RAG."""
import os
import google.generativeai as genai
from typing import List
import numpy as np


class GeminiEmbedder:
    """Wrapper for Gemini embedding model."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Gemini embedder.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('models/embedding-001')
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        try:
            result = self.model.embed_content(text)
            return result['embedding']
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

