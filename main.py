"""FastAPI backend for RAG-based policy Q&A."""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
from dotenv import load_dotenv
import shutil

from rag.embedder import GeminiEmbedder
from rag.retriever import FAISSRetriever
from rag.chunker import chunk_text
from rag.summarizer import GeminiSummarizer
from utils.pdf_reader import extract_text_from_pdf_bytes

# Load environment variables
load_dotenv()

app = FastAPI(title="Internal Policy Agent API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
embedder = None
retriever = None
summarizer = None


def initialize_components():
    """Initialize RAG components."""
    global embedder, retriever, summarizer
    
    try:
        embedder = GeminiEmbedder()
        retriever = FAISSRetriever(embedder)
        summarizer = GeminiSummarizer()
        
        # Try to load existing index
        retriever.load_index()
    except Exception as e:
        print(f"Warning: Could not initialize components: {e}")


# Initialize on startup
@app.on_event("startup")
async def startup_event():
    initialize_components()


# Request/Response models
class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]


class UploadResponse(BaseModel):
    message: str
    chunks_processed: int


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Internal Policy Agent API"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF policy document.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if embedder is None or retriever is None:
        raise HTTPException(status_code=500, detail="RAG components not initialized")
    
    try:
        # Save uploaded file
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract text from PDF
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        
        text = extract_text_from_pdf_bytes(pdf_bytes)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Chunk text
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
        
        # Add to index
        retriever.add_documents(chunks)
        
        return UploadResponse(
            message=f"Successfully processed {file.filename}",
            chunks_processed=len(chunks)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the policy documents.
    """
    if retriever is None or summarizer is None:
        raise HTTPException(status_code=500, detail="RAG components not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Retrieve relevant chunks
        results = retriever.search(request.question, k=5)
        
        if not results:
            return QuestionResponse(
                answer="No relevant information found in the policy documents. Please upload policy documents first.",
                sources=[]
            )
        
        # Extract context chunks
        context_chunks = [chunk for chunk, _ in results]
        
        # Generate response
        answer = summarizer.generate_response(request.question, context_chunks)
        
        # Return sources (first 3 most relevant)
        sources = context_chunks[:3]
        
        return QuestionResponse(
            answer=answer,
            sources=sources
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/status")
async def get_status():
    """Get API status and index information."""
    if retriever is None:
        return {
            "status": "not_initialized",
            "documents_indexed": 0
        }
    
    return {
        "status": "ready",
        "documents_indexed": len(retriever.documents) if retriever.documents else 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

