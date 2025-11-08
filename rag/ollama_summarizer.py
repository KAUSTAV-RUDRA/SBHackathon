"""Ollama-based summarization/answering for RAG responses."""
import os
from typing import List
import requests


class OllamaSummarizer:
    """Wrapper for Ollama text generation (local)."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        """
        Initialize Ollama summarizer.

        Args:
            base_url: Ollama server base URL (default: http://localhost:11434)
            model: Generation model name (e.g., 'llama3', 'qwen2', 'mistral')
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")

    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate an answer using query and retrieved context chunks via Ollama."""
        context_text = "\n\n".join([f"[Document {i+1}]\n{c}" for i, c in enumerate(context)])
        prompt = (
            "You are a helpful assistant that answers questions based on the provided policy documents.\n\n"
            f"Context from policy documents:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Provide a clear, accurate answer based only on the context. "
            "If the context does not contain enough information, say so explicitly."
        )

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "").strip()
            return text
        except Exception as e:
            raise Exception(f"Error generating response via Ollama: {str(e)}")


