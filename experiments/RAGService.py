import os
import pdfplumber
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import List, Optional, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import numpy as np
import faiss
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class RAGService:
    """
    RAG Service using FAISS for fast vector search with data ingestion capabilities.
    """
    
    def __init__(self, openai_api_key: str, index_path: str = "./faiss_index"):
        """
        Initialize RAG Service with FAISS and OpenAI credentials.
        
        Args:
            openai_api_key: API key for OpenAI
            index_path: Directory path to store FAISS index and metadata
        """
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # FAISS setup
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.index = None  # FAISS index
        self.metadata = []  # List of dicts with text, collection, chunk_index
        self.dimension = 1536  # OpenAI embedding dimension
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self):
        """Load FAISS index and metadata from disk if they exist."""
        index_file = self.index_path / "faiss.index"
        metadata_file = self.index_path / "metadata.pkl"
        
        if index_file.exists() and metadata_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"✓ Loaded FAISS index with {len(self.metadata)} vectors")
            except Exception as e:
                print(f"Warning: Could not load existing index: {e}")
                self._initialize_index()
        else:
            self._initialize_index()
    
    def _initialize_index(self):
        """Initialize a new FAISS index."""
        # Use IndexFlatIP for cosine similarity (after L2 normalization)
        # For better performance with large datasets, consider IndexIVFFlat
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        print("✓ Initialized new FAISS index")
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            index_file = self.index_path / "faiss.index"
            metadata_file = self.index_path / "metadata.pkl"
            
            faiss.write_index(self.index, str(index_file))
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            print(f"✓ Saved FAISS index with {len(self.metadata)} vectors")
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def data_ingestion_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF files using pdfplumber."""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def data_ingestion_websites(self, url: str) -> str:
        """Extract text from websites using BeautifulSoup."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            raise Exception(f"Error fetching website: {str(e)}")
    
    def data_ingestion_excel(self, excel_path: str) -> str:
        """Extract text from Excel files using pandas."""
        try:
            df = pd.read_excel(excel_path)
            text = df.to_string(index=False)
            return text
        except Exception as e:
            raise Exception(f"Error reading Excel: {str(e)}")
    
    def load_data(
        self,
        collection_name: str,
        url_link: Optional[str] = None,
        pdf_file: Optional[str] = None,
        excel_file: Optional[str] = None
    ):
        """
        Load data into FAISS index.
        
        Args:
            collection_name: Logical collection name for grouping
            url_link: URL to scrape (optional)
            pdf_file: Path to PDF file (optional)
            excel_file: Path to Excel file (optional)
        """
        try:
            # Extract text based on source type
            text = ""
            source_info = ""
            if url_link:
                text = self.data_ingestion_websites(url_link)
                source_info = f"URL: {url_link}"
            elif pdf_file:
                text = self.data_ingestion_pdf(pdf_file)
                source_info = f"PDF: {pdf_file}"
            elif excel_file:
                text = self.data_ingestion_excel(excel_file)
                source_info = f"Excel: {excel_file}"
            else:
                raise ValueError("At least one data source must be provided")
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Generate embeddings
            embeddings_list = []
            for chunk in chunks:
                embedding = self.embeddings.embed_query(chunk)
                embeddings_list.append(embedding)
            
            # Convert to numpy array and normalize for cosine similarity
            vectors = np.array(embeddings_list, dtype=np.float32)
            faiss.normalize_L2(vectors)  # Normalize for cosine similarity
            
            # Add to FAISS index
            self.index.add(vectors)
            
            # Store metadata
            for i, chunk in enumerate(chunks):
                self.metadata.append({
                    "text": chunk,
                    "collection": collection_name,
                    "chunk_index": i,
                    "source": source_info
                })
            
            # Save index
            self._save_index()
            
            print(f"✓ Loaded {len(chunks)} chunks from {source_info} to collection '{collection_name}'")
            return {"status": "success", "chunks_loaded": len(chunks)}
        
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    async def load_data_async(
        self,
        collection_name: str,
        url_links: Optional[List[str]] = None,
        pdf_files: Optional[List[str]] = None,
        excel_files: Optional[List[str]] = None
    ):
        """
        Load data from multiple sources asynchronously.
        
        Args:
            collection_name: Logical collection name for grouping
            url_links: List of URLs to scrape (optional)
            pdf_files: List of PDF file paths (optional)
            excel_files: List of Excel file paths (optional)
        """
        try:
            tasks = []
            source_types = []
            
            if url_links:
                for url in url_links:
                    tasks.append(self.async_data_ingestion_websites(url))
                    source_types.append(f"URL: {url}")
            
            if pdf_files:
                for pdf in pdf_files:
                    tasks.append(self.async_data_ingestion_pdf(pdf))
                    source_types.append(f"PDF: {pdf}")
            
            if excel_files:
                for excel in excel_files:
                    tasks.append(self.async_data_ingestion_excel(excel))
                    source_types.append(f"Excel: {excel}")
            
            if not tasks:
                raise ValueError("At least one data source must be provided")
            
            print(f"Starting parallel ingestion of {len(tasks)} sources...")
            texts = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_chunks = []
            all_embeddings = []
            successful_sources = []
            failed_sources = []
            
            for text, source_type in zip(texts, source_types):
                if isinstance(text, Exception):
                    print(f"Failed to ingest {source_type}: {str(text)}")
                    failed_sources.append({"source": source_type, "error": str(text)})
                    continue
                
                chunks = self.text_splitter.split_text(text)
                
                # Generate embeddings for these chunks
                for chunk in chunks:
                    embedding = self.embeddings.embed_query(chunk)
                    all_embeddings.append(embedding)
                    all_chunks.append({
                        "text": chunk,
                        "collection": collection_name,
                        "source": source_type
                    })
                
                successful_sources.append({"source": source_type, "chunks": len(chunks)})
                print(f"✓ Extracted {len(chunks)} chunks from {source_type}")
            
            if not all_chunks:
                raise Exception("No data extracted from any source")
            
            # Convert to numpy array and normalize
            vectors = np.array(all_embeddings, dtype=np.float32)
            faiss.normalize_L2(vectors)
            
            # Add to FAISS index
            self.index.add(vectors)
            
            # Add metadata with chunk indices
            for i, chunk_meta in enumerate(all_chunks):
                self.metadata.append({
                    "text": chunk_meta["text"],
                    "collection": chunk_meta["collection"],
                    "chunk_index": i,
                    "source": chunk_meta["source"]
                })
            
            # Save index
            self._save_index()
            
            print(f"✓ Loaded {len(all_chunks)} total chunks to collection '{collection_name}'")
            
            return {
                "status": "success",
                "total_chunks_loaded": len(all_chunks),
                "sources_processed": len(successful_sources),
                "sources_failed": len(failed_sources),
                "successful_sources": successful_sources,
                "failed_sources": failed_sources if failed_sources else None
            }
        
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    async def async_data_ingestion_pdf(self, pdf_path: str) -> str:
        """Async PDF ingestion."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.data_ingestion_pdf, pdf_path)
    
    async def async_data_ingestion_websites(self, url: str) -> str:
        """Async website ingestion."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.data_ingestion_websites, url)
    
    async def async_data_ingestion_excel(self, excel_path: str) -> str:
        """Async Excel ingestion."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.data_ingestion_excel, excel_path)
    
    def retrieval_based_search(
        self, 
        query: str, 
        collections: Optional[List[str]] = None, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using FAISS.
        
        Args:
            query: Search query
            collections: List of collection names to filter by (optional)
            top_k: Number of top results to return
            
        Returns:
            List of search results with text, score, collection, and chunk_index
        """
        try:
            if self.index.ntotal == 0:
                print("Warning: Index is empty")
                return []
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # Search FAISS index (get more results for filtering)
            search_k = min(top_k * 10, self.index.ntotal) if collections else top_k
            distances, indices = self.index.search(query_vector, search_k)
            
            # Prepare results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty results
                    continue
                
                meta = self.metadata[idx]
                
                # Filter by collection if specified
                if collections and meta["collection"] not in collections:
                    continue
                
                results.append({
                    "text": meta["text"],
                    "score": float(dist),  # Cosine similarity score
                    "collection": meta["collection"],
                    "chunk_index": meta["chunk_index"],
                    "source": meta.get("source", "unknown")
                })
                
                # Stop if we have enough results
                if len(results) >= top_k:
                    break
            
            return results
        
        except Exception as e:
            raise Exception(f"Error performing search: {str(e)}")
    
    def clear_index(self):
        """Clear the entire FAISS index and metadata."""
        self._initialize_index()
        self._save_index()
        print("✓ Cleared FAISS index")
    
    def delete_collection(self, collection_name: str):
        """
        Remove all vectors belonging to a specific collection.
        Note: FAISS doesn't support efficient deletion, so we rebuild the index.
        """
        try:
            # Filter out metadata for this collection
            remaining_metadata = [m for m in self.metadata if m["collection"] != collection_name]
            removed_count = len(self.metadata) - len(remaining_metadata)
            
            if removed_count == 0:
                print(f"No data found for collection '{collection_name}'")
                return
            
            # Rebuild index with remaining data
            self._initialize_index()
            
            if remaining_metadata:
                # Re-embed and add remaining chunks
                vectors = []
                for meta in remaining_metadata:
                    embedding = self.embeddings.embed_query(meta["text"])
                    vectors.append(embedding)
                
                vectors = np.array(vectors, dtype=np.float32)
                faiss.normalize_L2(vectors)
                self.index.add(vectors)
            
            self.metadata = remaining_metadata
            self._save_index()
            
            print(f"✓ Removed {removed_count} chunks from collection '{collection_name}'")
            
        except Exception as e:
            raise Exception(f"Error deleting collection: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        collections = {}
        for meta in self.metadata:
            coll = meta["collection"]
            collections[coll] = collections.get(coll, 0) + 1
        
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "collections": collections,
            "index_path": str(self.index_path)
        }