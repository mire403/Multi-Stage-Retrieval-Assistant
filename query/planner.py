"""
Query Planner - Generates execution plans based on query analysis.
"""

from typing import Dict, List, Optional, Any
from .analyzer import QueryPlan, QueryIntent


class QueryPlanner:
    """
    Plans the retrieval strategy based on query analysis.
    Determines which retrieval strategies to use and how to configure them.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Query Planner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    def plan(self, query_plan: QueryPlan) -> Dict[str, Any]:
        """
        Generate execution plan based on query analysis.
        
        Args:
            query_plan: Analyzed query plan
            
        Returns:
            Execution plan dictionary
        """
        execution_plan = {
            "stage1": self._plan_stage1(query_plan),
            "stage2": self._plan_stage2(query_plan),
            "stage3": self._plan_stage3(query_plan),
        }
        
        return execution_plan
    
    def _plan_stage1(self, query_plan: QueryPlan) -> Dict[str, Any]:
        """
        Plan Stage 1 (Broad Retrieval) strategy.
        
        Args:
            query_plan: Query plan
            
        Returns:
            Stage 1 configuration
        """
        base_config = self.config.get("stage1", {})
        
        # Adjust retrieval parameters based on query characteristics
        if query_plan.needs_multi_hop:
            # Multi-hop queries need larger candidate pool
            k = base_config.get("final_k", 200) * 1.5
        elif query_plan.intent == QueryIntent.FACTUAL:
            # Factual queries can be more focused
            k = base_config.get("final_k", 200) * 0.8
        else:
            k = base_config.get("final_k", 200)
        
        # Choose retrieval strategy
        if query_plan.intent == QueryIntent.COMPARATIVE:
            strategy = "hybrid"  # Need both keyword and semantic
        elif len(query_plan.key_concepts) > 5:
            strategy = "hybrid"  # Complex queries benefit from hybrid
        else:
            strategy = base_config.get("strategy", "hybrid")
        
        return {
            "strategy": strategy,
            "vector_k": int(base_config.get("vector_k", 200)),
            "keyword_k": int(base_config.get("keyword_k", 100)),
            "final_k": int(k),
            "min_score": base_config.get("min_score", 0.0),
            "use_key_concepts": True,
            "use_constraints": len(query_plan.constraints) > 0,
        }
    
    def _plan_stage2(self, query_plan: QueryPlan) -> Dict[str, Any]:
        """
        Plan Stage 2 (Semantic Refinement) strategy.
        
        Args:
            query_plan: Query plan
            
        Returns:
            Stage 2 configuration
        """
        base_config = self.config.get("stage2", {})
        
        # Adjust refinement parameters
        if query_plan.complexity_score > 0.7:
            max_contexts = base_config.get("max_contexts", 10) * 1.2
        else:
            max_contexts = base_config.get("max_contexts", 10)
        
        # Enable sub-question decomposition for complex queries
        enable_decomposition = (
            base_config.get("enable_subquestion_decomposition", True) and
            (query_plan.needs_multi_hop or query_plan.complexity_score > 0.6)
        )
        
        return {
            "use_llm_judge": base_config.get("use_llm_judge", True),
            "use_cross_encoder": base_config.get("use_cross_encoder", False),
            "max_contexts": int(max_contexts),
            "min_relevance_score": base_config.get("min_relevance_score", 0.5),
            "enable_subquestion_decomposition": enable_decomposition,
            "query_intent": query_plan.intent.value,
            "key_concepts": query_plan.key_concepts,
        }
    
    def _plan_stage3(self, query_plan: QueryPlan) -> Dict[str, Any]:
        """
        Plan Stage 3 (Answer Synthesis) strategy.
        
        Args:
            query_plan: Query plan
            
        Returns:
            Stage 3 configuration
        """
        base_config = self.config.get("stage3", {})
        
        # Adjust context window based on query complexity
        if query_plan.needs_multi_hop:
            max_tokens = base_config.get("max_context_tokens", 4000) * 1.5
        else:
            max_tokens = base_config.get("max_context_tokens", 4000)
        
        # Choose output format based on intent
        if query_plan.intent == QueryIntent.COMPARATIVE:
            output_format = "structured"  # Better for comparisons
        else:
            output_format = base_config.get("output_format", "structured")
        
        return {
            "max_context_tokens": int(max_tokens),
            "require_citations": base_config.get("require_citations", True),
            "output_format": output_format,
            "confidence_threshold": base_config.get("confidence_threshold", 0.6),
            "query_intent": query_plan.intent.value,
        }
