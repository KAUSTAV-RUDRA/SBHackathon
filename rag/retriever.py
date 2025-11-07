"""FAISS-based retrieval for RAG."""
import os
import numpy as np
import faiss
from typing import List, Tuple
from .embedder import GeminiEmbedder


class FAISSRetriever:
    """FAISS-based document retriever."""
    
    def __init__(self, embedder: GeminiEmbedder, index_path: str = "rag/index_store/faiss.index"):
        """
        Initialize FAISS retriever.
        
        Args:
            embedder: GeminiEmbedder instance
            index_path: Path to save/load FAISS index
        """
        self.embedder = embedder
        self.index_path = index_path
        self.index = None
        self.documents = []  # Store original documents for retrieval
        
    def build_index(self, texts: List[str]):
        """
        Build FAISS index from texts.
        
        Args:
            texts: List of text chunks to index
        """
        if not texts:
            raise ValueError("Cannot build index from empty text list")
        
        # Generate embeddings
        embeddings = self.embedder.embed_batch(texts)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)
        
        # Store documents
        self.documents = texts
        
        # Save index
        self.save_index()
    
    def save_index(self):
        """Save FAISS index and documents to disk."""
        if self.index is None:
            return
        
        # Save FAISS index
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        
        # Save documents metadata
        import json
        metadata_path = self.index_path.replace('.index', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def load_index(self):
        """Load FAISS index and documents from disk."""
        if not os.path.exists(self.index_path):
            return False
        
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Load documents metadata
        import json
        metadata_path = self.index_path.replace('.index', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
        
        return True
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (document, distance) tuples, sorted by relevance
        """
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, len(self.documents)))
        
        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(distances[0][i])))
        
        return results
    
    def add_documents(self, texts: List[str]):
        """
        Add new documents to existing index.
        
        Args:
            texts: List of new text chunks to add
        """
        if self.index is None:
            self.build_index(texts)
            return
        
        # Generate embeddings for new texts
        embeddings = self.embedder.embed_batch(texts)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Update documents
        self.documents.extend(texts)
        
        # Save updated index
        self.save_index()

