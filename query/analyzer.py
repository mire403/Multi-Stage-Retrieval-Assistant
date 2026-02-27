"""
Query Analyzer - Analyzes user queries to extract intent, constraints, and requirements.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import re


class QueryIntent(str, Enum):
    """Types of query intents."""
    FACTUAL = "factual"  # What is X?
    COMPARATIVE = "comparative"  # Compare X and Y
    PROCEDURAL = "procedural"  # How to do X?
    EXPLORATORY = "exploratory"  # Tell me about X
    CAUSAL = "causal"  # Why does X happen?
    TEMPORAL = "temporal"  # When did X happen?


class QueryPlan(BaseModel):
    """Structured query plan after analysis."""
    intent: QueryIntent = Field(..., description="Detected query intent")
    needs_multi_hop: bool = Field(default=False, description="Requires multi-hop reasoning")
    key_concepts: List[str] = Field(default_factory=list, description="Key concepts extracted")
    constraints: List[str] = Field(default_factory=list, description="Constraints (time, domain, etc.)")
    sub_questions: Optional[List[str]] = Field(default=None, description="Decomposed sub-questions")
    requires_numerical: bool = Field(default=False, description="Requires numerical data")
    requires_temporal: bool = Field(default=False, description="Requires temporal information")
    complexity_score: float = Field(default=0.5, description="Query complexity (0-1)")


class QueryAnalyzer:
    """
    Analyzes user queries to understand intent, extract key concepts,
    and identify constraints for the retrieval pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Query Analyzer.
        
        Args:
            config: Configuration dictionary with analysis settings
        """
        self.config = config or {}
        self.enable_intent_detection = self.config.get("enable_intent_detection", True)
        self.enable_constraint_extraction = self.config.get("enable_constraint_extraction", True)
        self.enable_multi_hop_detection = self.config.get("enable_multi_hop_detection", True)
        
        # Intent detection patterns
        self.intent_patterns = {
            QueryIntent.FACTUAL: [
                r"what (is|are|was|were)",
                r"define",
                r"explain (what|who|which)",
            ],
            QueryIntent.COMPARATIVE: [
                r"compare",
                r"difference between",
                r"vs\.|versus",
                r"better than",
                r"similar to",
            ],
            QueryIntent.PROCEDURAL: [
                r"how (to|do|can|should)",
                r"steps to",
                r"process of",
                r"way to",
            ],
            QueryIntent.EXPLORATORY: [
                r"tell me about",
                r"describe",
                r"overview of",
                r"information about",
            ],
            QueryIntent.CAUSAL: [
                r"why",
                r"what causes",
                r"reason for",
                r"because of",
            ],
            QueryIntent.TEMPORAL: [
                r"when",
                r"what year",
                r"what time",
                r"history of",
            ],
        }
        
        # Constraint patterns
        self.constraint_patterns = {
            "recent": [r"recent", r"latest", r"new", r"current", r"202[0-9]", r"202[0-9]"],
            "research": [r"research", r"study", r"paper", r"publication"],
            "numerical": [r"\d+", r"how many", r"percentage", r"rate"],
            "temporal": [r"before", r"after", r"during", r"since", r"until"],
        }
    
    def analyze(self, query: str) -> QueryPlan:
        """
        Analyze a user query and generate a query plan.
        
        Args:
            query: User query string
            
        Returns:
            QueryPlan object with analysis results
        """
        query_lower = query.lower()
        
        # Detect intent
        intent = self._detect_intent(query_lower) if self.enable_intent_detection else QueryIntent.EXPLORATORY
        
        # Extract key concepts
        key_concepts = self._extract_concepts(query)
        
        # Extract constraints
        constraints = self._extract_constraints(query_lower) if self.enable_constraint_extraction else []
        
        # Detect multi-hop requirement
        needs_multi_hop = self._detect_multi_hop(query) if self.enable_multi_hop_detection else False
        
        # Detect requirements
        requires_numerical = self._requires_numerical(query_lower)
        requires_temporal = self._requires_temporal(query_lower)
        
        # Calculate complexity
        complexity_score = self._calculate_complexity(query, key_concepts, needs_multi_hop)
        
        return QueryPlan(
            intent=intent,
            needs_multi_hop=needs_multi_hop,
            key_concepts=key_concepts,
            constraints=constraints,
            requires_numerical=requires_numerical,
            requires_temporal=requires_temporal,
            complexity_score=complexity_score,
        )
    
    def _detect_intent(self, query_lower: str) -> QueryIntent:
        """Detect query intent using pattern matching."""
        scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query_lower))
            if score > 0:
                scores[intent] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return QueryIntent.EXPLORATORY
    
    def _extract_concepts(self, query: str) -> List[str]:
        """
        Extract key concepts from the query.
        Simple implementation - can be enhanced with NER or LLM.
        """
        # Remove common stop words and question words
        stop_words = {
            "what", "is", "are", "was", "were", "the", "a", "an",
            "how", "why", "when", "where", "who", "which", "do",
            "does", "did", "can", "could", "should", "would",
            "tell", "me", "about", "explain", "describe", "compare",
        }
        
        # Simple tokenization and filtering
        words = re.findall(r'\b\w+\b', query.lower())
        concepts = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for concept in concepts:
            if concept not in seen:
                seen.add(concept)
                unique_concepts.append(concept)
        
        return unique_concepts[:10]  # Limit to top 10
    
    def _extract_constraints(self, query_lower: str) -> List[str]:
        """Extract constraints from the query."""
        constraints = []
        
        for constraint_type, patterns in self.constraint_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                constraints.append(constraint_type)
        
        return constraints
    
    def _detect_multi_hop(self, query: str) -> bool:
        """
        Detect if query requires multi-hop reasoning.
        Simple heuristics - can be enhanced with LLM.
        """
        multi_hop_indicators = [
            "and then",
            "after that",
            "related to",
            "connected to",
            "based on",
            "according to",
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in multi_hop_indicators)
    
    def _requires_numerical(self, query_lower: str) -> bool:
        """Check if query requires numerical data."""
        return "numerical" in self._extract_constraints(query_lower) or \
               bool(re.search(r"\d+", query_lower))
    
    def _requires_temporal(self, query_lower: str) -> bool:
        """Check if query requires temporal information."""
        return "temporal" in self._extract_constraints(query_lower) or \
               self._detect_intent(query_lower) == QueryIntent.TEMPORAL
    
    def _calculate_complexity(self, query: str, concepts: List[str], multi_hop: bool) -> float:
        """
        Calculate query complexity score (0-1).
        Higher score = more complex.
        """
        base_complexity = min(len(query.split()) / 50.0, 1.0)
        concept_complexity = min(len(concepts) / 10.0, 1.0)
        multi_hop_boost = 0.3 if multi_hop else 0.0
        
        return min(base_complexity * 0.4 + concept_complexity * 0.3 + multi_hop_boost, 1.0)
