"""PDF text extraction utility."""
import PyPDF2
from typing import List, Optional


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")
    
    return text


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text content from PDF bytes.
    
    Args:
        pdf_bytes: PDF file content as bytes
        
    Returns:
        Extracted text as a string
    """
    text = ""
    try:
        from io import BytesIO
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")
    
    return text

