"""Streamlit UI for the Customer Support RAG system."""
import streamlit as st
import requests
import json
from typing import Dict, Any, List

# API Configuration
API_BASE_URL = "http://localhost:8000"


def query_rag(query: str, k: int = 5) -> Dict[str, Any]:
    """Query the RAG API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json={"query": query, "k": k},
            timeout=100
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying API: {e}")
        return {}


def get_stats() -> Dict[str, Any]:
    """Get vector store statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"Could not fetch stats: {e}")
        return {}


def process_document(file_path: str) -> bool:
    """Process a document via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/process_document",
            json={"file_path": file_path},
            timeout=300
        )
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return False


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Customer Support RAG Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Customer Support RAG Assistant")
    st.markdown("Ask questions about your documents and get AI-powered answers with confidence scores.")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Stats
        stats = get_stats()
        if stats:
            st.metric("Total Chunks", stats.get("total_chunks", 0))
            st.metric("Embedding Model", stats.get("embedding_model", "N/A"))
            st.metric("Embedding Dimension", stats.get("embedding_dim", 0))
        
        st.divider()
        
        st.header("⚙️ Settings")
        k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5)
        show_chunks = st.checkbox("Show retrieved chunks", value=True)
        show_validation = st.checkbox("Show validation details", value=True)
    
    # Main content
    tab1, tab2 = st.tabs(["💬 Query", "📄 Process Documents"])
    
    with tab1:
        st.header("Ask a Question")
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What is the refund policy?"
        )
        
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query:
                with st.spinner("Searching and generating answer..."):
                    result = query_rag(query, k=k)
                    
                    if result:
                        # Display answer
                        st.subheader("📝 Answer")
                        st.write(result.get("answer", "No answer generated"))
                        
                        # Confidence score
                        confidence = result.get("confidence_score", 0.0)
                        confidence_color = "green" if confidence >= 0.7 else "orange" if confidence >= 0.5 else "red"
                        st.markdown(f"**Confidence Score:** :{confidence_color}[{confidence:.2%}]")
                        
                        # Validation details
                        if show_validation:
                            validation = result.get("validation", {})
                            with st.expander("🔍 Validation Details"):
                                st.json(validation)
                        
                        # Retrieved chunks
                        if show_chunks:
                            st.subheader("📚 Retrieved Chunks")
                            retrieved_chunks = result.get("retrieved_chunks", [])
                            
                            if retrieved_chunks:
                                for i, chunk in enumerate(retrieved_chunks, 1):
                                    with st.expander(f"Chunk {i} (Similarity: {chunk.get('similarity_score', 0):.3f})"):
                                        st.write(chunk.get("content", ""))
                                        
                                        # Metadata
                                        metadata = chunk.get("metadata", {})
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.caption(f"**File:** {metadata.get('file_name', 'N/A')}")
                                        with col2:
                                            st.caption(f"**Type:** {metadata.get('chunk_type', 'N/A')}")
                                        with col3:
                                            st.caption(f"**Page:** {metadata.get('page', 'N/A')}")
                            else:
                                st.info("No chunks retrieved")
            else:
                st.warning("Please enter a question")
    
    with tab2:
        st.header("Process Documents")
        st.markdown("Add documents to the knowledge base. Documents will be automatically chunked and indexed.")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a document (PDF or Word)",
            type=["pdf", "docx", "doc"]
        )
        
        if uploaded_file:
            # Save uploaded file
            import os
            from pathlib import Path
            from config.config import DOCUMENTS_DIR
            
            file_path = Path(DOCUMENTS_DIR) / uploaded_file.name
            
            if st.button("📤 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    # Save file
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process via API
                    success = process_document(str(file_path))
                    
                    if success:
                        st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
                        st.balloons()
                    else:
                        st.error(f"❌ Failed to process document '{uploaded_file.name}'")
        
        # Manual file path input
        st.divider()
        st.subheader("Or process from file path")
        manual_path = st.text_input("Enter file path:")
        if st.button("Process from Path") and manual_path:
            with st.spinner("Processing document..."):
                success = process_document(manual_path)
                if success:
                    st.success(f"✅ Document processed successfully!")
                else:
                    st.error(f"❌ Failed to process document")


if __name__ == "__main__":
    main()
