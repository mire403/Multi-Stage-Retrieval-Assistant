"""
Ranker - Re-ranking utilities for Stage 2 refinement.
"""

from typing import List, Dict, Optional, Any
from .stage1_broad import Candidate


class Ranker:
    """
    Re-ranking utilities for semantic refinement.
    Supports cross-encoder models and LLM-based ranking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Ranker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.use_cross_encoder = self.config.get("use_cross_encoder", False)
        self.cross_encoder_model = None
        
        if self.use_cross_encoder:
            self._init_cross_encoder()
    
    def _init_cross_encoder(self):
        """Initialize cross-encoder model for re-ranking."""
        try:
            from sentence_transformers import CrossEncoder
            
            model_name = self.config.get(
                "cross_encoder_model",
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            self.cross_encoder_model = CrossEncoder(model_name)
        except ImportError:
            print("Warning: sentence-transformers not available for cross-encoder")
            self.use_cross_encoder = False
        except Exception as e:
            print(f"Warning: Could not load cross-encoder: {e}")
            self.use_cross_encoder = False
    
    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        top_k: Optional[int] = None
    ) -> List[Candidate]:
        """
        Re-rank candidates using cross-encoder.
        
        Args:
            query: User query
            candidates: List of candidates to re-rank
            top_k: Number of top results to return
            
        Returns:
            Re-ranked candidates
        """
        if not self.use_cross_encoder or not self.cross_encoder_model:
            # Return original order if cross-encoder not available
            return candidates[:top_k] if top_k else candidates
        
        if not candidates:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [
            [query, candidate.content or ""]
            for candidate in candidates
        ]
        
        # Get scores from cross-encoder
        scores = self.cross_encoder_model.predict(pairs)
        
        # Update candidate scores
        reranked = []
        for candidate, score in zip(candidates, scores):
            # Combine original score with cross-encoder score
            combined_score = (candidate.score * 0.3 + float(score) * 0.7)
            
            reranked.append(Candidate(
                doc_id=candidate.doc_id,
                score=combined_score,
                source=candidate.source,
                content=candidate.content,
                metadata=candidate.metadata,
            ))
        
        # Sort by new score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        return reranked[:top_k] if top_k else reranked
