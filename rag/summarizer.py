"""Gemini-based summarization for RAG responses."""
import os
import google.generativeai as genai
from typing import List


class GeminiSummarizer:
    """Wrapper for Gemini summarization."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Gemini summarizer.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
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
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
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
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Error generating summary: {str(e)}")

