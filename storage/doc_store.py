"""
Document Store - Manages document metadata and content.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import json
import os
from pathlib import Path


class Document(BaseModel):
    """Document model."""
    doc_id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    chunk_id: Optional[str] = Field(default=None, description="Chunk identifier if this is a chunk")
    parent_doc_id: Optional[str] = Field(default=None, description="Parent document ID if this is a chunk")


class DocumentStore:
    """
    Manages document storage and retrieval.
    Supports in-memory and file-based storage.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Document Store.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.store_type = self.config.get("type", "memory")
        self.path = self.config.get("path", "./data/doc_store.json")
        
        if self.store_type == "memory":
            self.documents: Dict[str, Document] = {}
        elif self.store_type == "json":
            self._ensure_directory()
            self._load_from_file()
        else:
            raise ValueError(f"Unsupported store type: {self.store_type}")
    
    def _ensure_directory(self):
        """Ensure the directory for the document store exists."""
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
    
    def _load_from_file(self):
        """Load documents from JSON file."""
        self.documents: Dict[str, Document] = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for doc_id, doc_data in data.items():
                        self.documents[doc_id] = Document(**doc_data)
            except Exception as e:
                print(f"Warning: Could not load document store: {e}")
    
    def _save_to_file(self):
        """Save documents to JSON file."""
        if self.store_type == "json":
            try:
                data = {
                    doc_id: doc.dict() for doc_id, doc in self.documents.items()
                }
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Warning: Could not save document store: {e}")
    
    def add_document(self, document: Document) -> None:
        """
        Add a document to the store.
        
        Args:
            document: Document to add
        """
        self.documents[document.doc_id] = document
        if self.store_type == "json":
            self._save_to_file()
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add multiple documents to the store.
        
        Args:
            documents: List of documents to add
        """
        for doc in documents:
            self.add_document(doc)
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document if found, None otherwise
        """
        return self.documents.get(doc_id)
    
    def get_documents(self, doc_ids: List[str]) -> List[Document]:
        """
        Retrieve multiple documents by IDs.
        
        Args:
            doc_ids: List of document identifiers
            
        Returns:
            List of documents (may contain None for missing documents)
        """
        return [self.get_document(doc_id) for doc_id in doc_ids]
    
    def get_all_documents(self) -> List[Document]:
        """
        Get all documents in the store.
        
        Returns:
            List of all documents
        """
        return list(self.documents.values())
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any]) -> List[Document]:
        """
        Search documents by metadata.
        
        Args:
            metadata_filter: Metadata key-value pairs to filter by
            
        Returns:
            List of matching documents
        """
        results = []
        for doc in self.documents.values():
            match = True
            for key, value in metadata_filter.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            if match:
                results.append(doc)
        return results
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the store.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if deleted, False if not found
        """
        if doc_id in self.documents:
            del self.documents[doc_id]
            if self.store_type == "json":
                self._save_to_file()
            return True
        return False
    
    def count(self) -> int:
        """
        Get the total number of documents.
        
        Returns:
            Document count
        """
        return len(self.documents)
