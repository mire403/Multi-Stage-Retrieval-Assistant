"""
Stage 1: Broad Retrieval - High recall, low precision candidate gathering.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
import re


class Candidate(BaseModel):
    """Candidate document from Stage 1."""
    doc_id: str = Field(..., description="Document identifier")
    score: float = Field(..., description="Retrieval score")
    source: str = Field(..., description="Source of retrieval (vector, keyword, metadata)")
    content: Optional[str] = Field(default=None, description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")


class Stage1BroadRetriever:
    """
    Stage 1 Broad Retriever - High recall retrieval using multiple strategies.
    
    Goal: Don't miss potentially relevant documents.
    Strategies: Vector search, keyword/BM25, metadata filtering.
    """
    
    def __init__(
        self,
        vector_store,
        doc_store,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Stage 1 Broad Retriever.
        
        Args:
            vector_store: VectorStore instance
            doc_store: DocumentStore instance
            config: Configuration dictionary
        """
        self.vector_store = vector_store
        self.doc_store = doc_store
        self.config = config or {}
        
        self.strategy = self.config.get("strategy", "hybrid")
        self.vector_k = self.config.get("vector_k", 200)
        self.keyword_k = self.config.get("keyword_k", 100)
        self.final_k = self.config.get("final_k", 200)
        self.min_score = self.config.get("min_score", 0.0)
    
    def retrieve(
        self,
        query: str,
        key_concepts: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        execution_plan: Optional[Dict[str, Any]] = None
    ) -> List[Candidate]:
        """
        Perform broad retrieval to gather candidates.
        
        Args:
            query: User query
            key_concepts: Key concepts from query analysis
            constraints: Constraints from query analysis
            execution_plan: Stage 1 execution plan
            
        Returns:
            List of candidate documents
        """
        if execution_plan:
            # Override config with execution plan
            self.strategy = execution_plan.get("strategy", self.strategy)
            self.vector_k = execution_plan.get("vector_k", self.vector_k)
            self.keyword_k = execution_plan.get("keyword_k", self.keyword_k)
            self.final_k = execution_plan.get("final_k", self.final_k)
        
        all_candidates = []
        
        # Strategy 1: Vector search (semantic similarity)
        if self.strategy in ["hybrid", "vector"]:
            vector_candidates = self._vector_search(query, self.vector_k)
            all_candidates.extend(vector_candidates)
        
        # Strategy 2: Keyword/BM25 search
        if self.strategy in ["hybrid", "keyword"]:
            keyword_candidates = self._keyword_search(query, self.keyword_k)
            all_candidates.extend(keyword_candidates)
        
        # Strategy 3: Metadata filtering (if constraints provided)
        if constraints and execution_plan and execution_plan.get("use_constraints", False):
            metadata_candidates = self._metadata_search(constraints)
            all_candidates.extend(metadata_candidates)
        
        # Deduplicate and merge scores
        merged_candidates = self._merge_candidates(all_candidates)
        
        # Filter by minimum score
        filtered_candidates = [
            c for c in merged_candidates
            if c.score >= self.min_score
        ]
        
        # Sort by score and return top k
        sorted_candidates = sorted(
            filtered_candidates,
            key=lambda x: x.score,
            reverse=True
        )
        
        return sorted_candidates[:self.final_k]
    
    def _vector_search(self, query: str, k: int) -> List[Candidate]:
        """Perform vector similarity search."""
        results = self.vector_store.search(query, k=k)
        
        candidates = []
        for result in results:
            candidates.append(Candidate(
                doc_id=result.doc_id,
                score=result.score,
                source="vector",
                content=result.content,
                metadata=result.metadata,
            ))
        
        return candidates
    
    def _keyword_search(self, query: str, k: int) -> List[Candidate]:
        """Perform keyword/BM25 search."""
        # Get all documents for BM25
        all_docs = self.doc_store.get_all_documents()
        
        if not all_docs:
            return []
        
        # Tokenize documents and query
        tokenized_docs = [self._tokenize(doc.content) for doc in all_docs]
        tokenized_query = self._tokenize(query)
        
        # Build BM25 index
        bm25 = BM25Okapi(tokenized_docs)
        
        # Get scores
        scores = bm25.get_scores(tokenized_query)
        
        # Create candidates
        doc_scores = list(zip(all_docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        candidates = []
        for doc, score in doc_scores[:k]:
            # Normalize BM25 score to 0-1 range (rough approximation)
            normalized_score = min(score / 10.0, 1.0) if score > 0 else 0.0
            
            candidates.append(Candidate(
                doc_id=doc.doc_id,
                score=normalized_score,
                source="keyword",
                content=doc.content,
                metadata=doc.metadata,
            ))
        
        return candidates
    
    def _metadata_search(self, constraints: List[str]) -> List[Candidate]:
        """Search by metadata constraints."""
        candidates = []
        
        # Simple implementation - can be enhanced
        for constraint in constraints:
            if constraint == "recent":
                # Filter by date if available in metadata
                docs = self.doc_store.search_by_metadata({"recent": True})
            elif constraint == "research":
                docs = self.doc_store.search_by_metadata({"type": "research"})
            else:
                continue
            
            for doc in docs:
                candidates.append(Candidate(
                    doc_id=doc.doc_id,
                    score=0.5,  # Default score for metadata matches
                    source="metadata",
                    content=doc.content,
                    metadata=doc.metadata,
                ))
        
        return candidates
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Convert to lowercase and split on non-word characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _merge_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Merge candidates from different sources, combining scores.
        
        Args:
            candidates: List of candidates (may have duplicates)
            
        Returns:
            Deduplicated and merged candidates
        """
        candidate_dict: Dict[str, Candidate] = {}
        
        for candidate in candidates:
            doc_id = candidate.doc_id
            
            if doc_id in candidate_dict:
                # Merge: take max score and combine sources
                existing = candidate_dict[doc_id]
                # Boost score if found in multiple sources
                if existing.source != candidate.source:
                    merged_score = min(existing.score + candidate.score * 0.3, 1.0)
                else:
                    merged_score = max(existing.score, candidate.score)
                
                candidate_dict[doc_id] = Candidate(
                    doc_id=doc_id,
                    score=merged_score,
                    source=f"{existing.source}+{candidate.source}",
                    content=existing.content or candidate.content,
                    metadata=existing.metadata or candidate.metadata,
                )
            else:
                candidate_dict[doc_id] = candidate
        
        return list(candidate_dict.values())
