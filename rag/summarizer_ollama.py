"""Ollama-based summarization for RAG responses."""
import os
import requests
from typing import List

class OllamaSummarizer:
    """Wrapper for Ollama summarization."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        """
        Initialize Ollama summarizer.
        
        Args:
            base_url: Ollama API base URL
            model: Model name to use (e.g., 'llama2', 'mistral', etc.)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/generate")
            if response.status_code != 200:
                raise ValueError(f"Ollama API not available at {base_url}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Could not connect to Ollama API: {str(e)}")
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """
        Generate a response using query and retrieved context.
        
        Args:
            query: User question
            context: List of relevant document chunks
            
        Returns:
            Generated response
        """
        # Combine context
        context_text = "\n\n".join([f"[Document {i+1}]\n{chunk}" for i, chunk in enumerate(context)])
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided policy documents.

Context from policy documents:
{context_text}

Question: {query}

Please provide a clear, accurate answer based on the context above. If the context doesn't contain enough information to answer the question, say so explicitly."""
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get('response', '')
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """
        Summarize a text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        prompt = f"""Please provide a concise summary of the following text in {max_length} words or less:

{text}

Summary:"""
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get('response', '')
        except Exception as e:
            raise Exception(f"Error generating summary: {str(e)}")