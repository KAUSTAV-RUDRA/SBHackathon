"""Streamlit frontend for Policy Q&A."""
import streamlit as st
import requests
import os
from typing import Optional

# API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000")


def main():
    st.set_page_config(
        page_title="Internal Policy Agent",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("📋 Internal Policy Agent")
    st.markdown("Ask questions about your policy documents")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📤 Upload Policy Documents")
        
        uploaded_file = st.file_uploader(
            "Upload a PDF policy document",
            type=["pdf"],
            help="Upload PDF files containing policy information"
        )
        
        if uploaded_file is not None:
            if st.button("Upload and Process"):
                with st.spinner("Processing PDF..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        response = requests.post(f"{API_URL}/upload", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ {result['message']}")
                            st.info(f"Processed {result['chunks_processed']} text chunks")
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error uploading file: {str(e)}")
        
        st.divider()
        
        # Status check
        if st.button("Check Status"):
            try:
                response = requests.get(f"{API_URL}/status")
                if response.status_code == 200:
                    status = response.json()
                    st.info(f"📊 Documents indexed: {status.get('documents_indexed', 0)}")
            except Exception as e:
                st.warning(f"Could not connect to API: {str(e)}")
    
    # Main Q&A interface
    st.header("💬 Ask a Question")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.text_area(f"Source {i}", source, height=100, disabled=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the policy documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"question": prompt}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get("answer", "No answer provided")
                        sources = result.get("sources", [])
                        
                        st.markdown(answer)
                        
                        if sources:
                            with st.expander("📚 Sources"):
                                for i, source in enumerate(sources, 1):
                                    st.text_area(f"Source {i}", source, height=100, disabled=True)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        error_msg = response.json().get("detail", "Unknown error")
                        st.error(f"Error: {error_msg}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error: {error_msg}"
                        })
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to the API. Make sure the FastAPI server is running.")
                    st.info("Start the server with: `python main.py` or `uvicorn main:app --reload`")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {str(e)}"
                    })


if __name__ == "__main__":
    main()

