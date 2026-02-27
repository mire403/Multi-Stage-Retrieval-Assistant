"""
Vector Store - Manages vector embeddings and similarity search.
"""

from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
import os


class SearchResult(BaseModel):
    """Search result model."""
    doc_id: str
    score: float
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorStore:
    """
    Manages vector embeddings and provides similarity search.
    Supports ChromaDB and FAISS backends.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Vector Store.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.store_type = self.config.get("type", "chroma")
        self.persist_directory = self.config.get("persist_directory", "./data/vector_store")
        self.collection_name = self.config.get("collection_name", "documents")
        
        # Initialize embedding model
        embedding_config = self.config.get("embeddings", {})
        model_name = embedding_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize vector store backend
        if self.store_type == "chroma":
            self._init_chroma()
        elif self.store_type == "faiss":
            self._init_faiss()
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
    
    def _init_chroma(self):
        """Initialize ChromaDB backend."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            os.makedirs(self.persist_directory, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
        except ImportError:
            raise ImportError("ChromaDB is required. Install with: pip install chromadb")
    
    def _init_faiss(self):
        """Initialize FAISS backend."""
        try:
            import faiss
            
            self.index = None
            self.doc_ids = []
            self.doc_contents = []
            self.faiss_path = os.path.join(self.persist_directory, "faiss.index")
            self.metadata_path = os.path.join(self.persist_directory, "faiss_metadata.json")
            
            # Load existing index if available
            if os.path.exists(self.faiss_path):
                self.index = faiss.read_index(self.faiss_path)
                import json
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, "r") as f:
                        metadata = json.load(f)
                        self.doc_ids = metadata.get("doc_ids", [])
                        self.doc_contents = metadata.get("doc_contents", [])
        except ImportError:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents with 'doc_id', 'content', and optional 'metadata'
        """
        if not documents:
            return
        
        # Extract texts for embedding
        texts = [doc["content"] for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True
        )
        
        if self.store_type == "chroma":
            self._add_to_chroma(documents, embeddings)
        elif self.store_type == "faiss":
            self._add_to_faiss(documents, embeddings)
    
    def _add_to_chroma(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents to ChromaDB."""
        ids = [doc["doc_id"] for doc in documents]
        texts = [doc["content"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        
        # Convert numpy array to list
        embeddings_list = embeddings.tolist()
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas
        )
    
    def _add_to_faiss(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents to FAISS."""
        import faiss
        import json
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index if it doesn't exist
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Add to index
        self.index.add(embeddings.astype("float32"))
        
        # Store metadata
        for doc in documents:
            self.doc_ids.append(doc["doc_id"])
            self.doc_contents.append(doc["content"])
        
        # Save index and metadata
        os.makedirs(os.path.dirname(self.faiss_path), exist_ok=True)
        faiss.write_index(self.index, self.faiss_path)
        
        metadata = {
            "doc_ids": self.doc_ids,
            "doc_contents": self.doc_contents
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f)
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: Query string
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        )
        
        if self.store_type == "chroma":
            return self._search_chroma(query_embedding, k, filter_metadata)
        elif self.store_type == "faiss":
            return self._search_faiss(query_embedding, k)
    
    def _search_chroma(
        self,
        query_embedding: np.ndarray,
        k: int,
        filter_metadata: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Search in ChromaDB."""
        where = filter_metadata if filter_metadata else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where
        )
        
        search_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                search_results.append(SearchResult(
                    doc_id=results["ids"][0][i],
                    score=1.0 - results["distances"][0][i] if "distances" in results else 0.0,
                    content=results["documents"][0][i] if "documents" in results else None,
                    metadata=results["metadatas"][0][i] if "metadatas" in results else None,
                ))
        
        return search_results
    
    def _search_faiss(self, query_embedding: np.ndarray, k: int) -> List[SearchResult]:
        """Search in FAISS."""
        import faiss
        
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        search_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.doc_ids):
                search_results.append(SearchResult(
                    doc_id=self.doc_ids[idx],
                    score=float(scores[0][i]),
                    content=self.doc_contents[idx] if idx < len(self.doc_contents) else None,
                ))
        
        return search_results
    
    def delete_documents(self, doc_ids: List[str]) -> None:
        """
        Delete documents from the vector store.
        
        Args:
            doc_ids: List of document IDs to delete
        """
        if self.store_type == "chroma":
            self.collection.delete(ids=doc_ids)
        elif self.store_type == "faiss":
            # FAISS doesn't support deletion easily, would need to rebuild index
            # For now, we'll mark them as deleted in metadata
            pass  # Implementation would require rebuilding index
