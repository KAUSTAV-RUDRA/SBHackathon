# Internal Policy Agent

A RAG (Retrieval-Augmented Generation) based Q&A system for internal policy documents. This application allows you to upload policy PDFs and ask questions about them using Google's Gemini AI.

## Features

- 📄 **PDF Upload**: Upload policy documents in PDF format
- 🔍 **Semantic Search**: FAISS-based vector search for relevant document chunks
- 🤖 **AI-Powered Q&A**: Gemini AI generates accurate answers based on retrieved context
- 💬 **Interactive UI**: Streamlit-based chat interface
- 🚀 **FastAPI Backend**: RESTful API for document processing and Q&A

## Architecture

- **Frontend**: Streamlit (`app.py`) - Interactive Q&A interface
- **Backend**: FastAPI (`main.py`) - RAG logic and API endpoints
- **RAG Pipeline**:
  - `chunker.py`: Text chunking utilities
  - `embedder.py`: Gemini embeddings generation
  - `retriever.py`: FAISS-based document retrieval
  - `summarizer.py`: Gemini-based response generation
- **Utils**: PDF text extraction (`utils/pdf_reader.py`)

## Setup

### 1. Prerequisites

- Python 3.8 or higher
- Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### 2. Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

1. Copy `.env` file and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

2. Update `API_URL` in `.env` if running on a different host/port.

### 4. Running the Application

#### Start the FastAPI Backend

```bash
# Option 1: Using uvicorn directly
uvicorn main:app --reload

# Option 2: Using Python
python main.py
```

The API will be available at `http://localhost:8000`

#### Start the Streamlit Frontend

In a new terminal:

```bash
streamlit run app.py
```

The UI will open in your browser at `http://localhost:8501`

## Usage

1. **Upload Documents**: Use the sidebar in the Streamlit app to upload PDF policy documents
2. **Ask Questions**: Type your question in the chat interface
3. **View Sources**: Expand the "Sources" section to see which document chunks were used to generate the answer

## API Endpoints

### `POST /upload`
Upload a PDF policy document.

**Request**: Multipart form data with `file` field
**Response**: 
```json
{
  "message": "Successfully processed document.pdf",
  "chunks_processed": 42
}
```

### `POST /ask`
Ask a question about the policy documents.

**Request**:
```json
{
  "question": "What is the company's remote work policy?"
}
```

**Response**:
```json
{
  "answer": "According to the policy documents...",
  "sources": ["[Document 1]\n...", "[Document 2]\n..."]
}
```

### `GET /status`
Get API status and number of indexed documents.

**Response**:
```json
{
  "status": "ready",
  "documents_indexed": 42
}
```

## Project Structure

```
internal-policy-agent/
├── app.py                  # Streamlit frontend (Q&A UI)
├── main.py                 # FastAPI backend (RAG logic)
├── input.json              # Input schema definition
├── output.json             # Output schema definition
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (GEMINI_API_KEY)
├── rag/
│   ├── chunker.py          # PDF → text → chunks
│   ├── embedder.py         # Gemini embeddings
│   ├── retriever.py        # FAISS search logic
│   ├── summarizer.py       # Gemini summarization
│   └── index_store/        # FAISS index files
├── utils/
│   └── pdf_reader.py       # Extract text from uploaded PDFs
├── uploads/                # Uploaded policy PDFs
└── README.md               # This file
```

## Troubleshooting

### API Connection Error
- Make sure the FastAPI backend is running on port 8000
- Check that `API_URL` in `.env` matches your backend URL

### Gemini API Errors
- Verify your `GEMINI_API_KEY` is correct in `.env`
- Check your API quota and billing status

### PDF Processing Errors
- Ensure PDFs are not password-protected
- Check that PDFs contain extractable text (not just images)

## License

MIT

