"""
Stage 2: Semantic Refinement - High precision filtering and re-ranking.
"""

from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field
from .stage1_broad import Candidate
from .ranker import Ranker


class RefinedContext(BaseModel):
    """Refined context from Stage 2."""
    doc_id: str = Field(..., description="Document identifier")
    content: str = Field(..., description="Document content")
    relevance_score: float = Field(..., description="Relevance score (0-1)")
    reason: str = Field(..., description="Reason for selection/rejection")
    sub_question_match: Optional[str] = Field(default=None, description="Matched sub-question if applicable")


class Stage2Refiner:
    """
    Stage 2 Semantic Refiner - High precision filtering.
    
    Goal: Remove false positives, keep only truly relevant contexts.
    Methods: LLM relevance judgment, sub-question decomposition, re-ranking.
    """
    
    def __init__(
        self,
        llm_client,
        ranker: Optional[Ranker] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Stage 2 Refiner.
        
        Args:
            llm_client: LLM client for relevance judgment
            ranker: Optional Ranker instance for re-ranking
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.ranker = ranker or Ranker(config.get("stage2", {}) if config else {})
        self.config = config or {}
        
        stage2_config = self.config.get("stage2", {})
        self.use_llm_judge = stage2_config.get("use_llm_judge", True)
        self.max_contexts = stage2_config.get("max_contexts", 10)
        self.min_relevance_score = stage2_config.get("min_relevance_score", 0.5)
        self.enable_subquestion_decomposition = stage2_config.get(
            "enable_subquestion_decomposition",
            True
        )
    
    def refine(
        self,
        query: str,
        candidates: List[Candidate],
        query_plan: Optional[Dict[str, Any]] = None,
        execution_plan: Optional[Dict[str, Any]] = None
    ) -> List[RefinedContext]:
        """
        Refine candidates to high-confidence contexts.
        
        Args:
            query: Original user query
            candidates: Candidates from Stage 1
            query_plan: Query plan from analyzer
            execution_plan: Stage 2 execution plan
            
        Returns:
            List of refined contexts
        """
        if execution_plan:
            # Override config with execution plan
            self.max_contexts = execution_plan.get("max_contexts", self.max_contexts)
            self.min_relevance_score = execution_plan.get(
                "min_relevance_score",
                self.min_relevance_score
            )
            self.enable_subquestion_decomposition = execution_plan.get(
                "enable_subquestion_decomposition",
                self.enable_subquestion_decomposition
            )
        
        if not candidates:
            return []
        
        # Step 1: Re-rank using cross-encoder if available
        if self.ranker:
            candidates = self.ranker.rerank(query, candidates, top_k=min(len(candidates), self.max_contexts * 3))
        
        # Step 2: Sub-question decomposition (if enabled and query is complex)
        sub_questions = None
        if self.enable_subquestion_decomposition and query_plan:
            if query_plan.get("needs_multi_hop", False) or query_plan.get("complexity_score", 0) > 0.6:
                sub_questions = self._decompose_subquestions(query, query_plan)
        
        # Step 3: LLM relevance judgment
        if self.use_llm_judge:
            refined_contexts = self._llm_relevance_judge(
                query,
                candidates[:self.max_contexts * 2],  # Judge top candidates
                sub_questions
            )
        else:
            # Fallback: use score-based filtering
            refined_contexts = self._score_based_filter(candidates)
        
        # Step 4: Filter by minimum relevance score
        filtered_contexts = [
            ctx for ctx in refined_contexts
            if ctx.relevance_score >= self.min_relevance_score
        ]
        
        # Step 5: Return top contexts
        return filtered_contexts[:self.max_contexts]
    
    def _decompose_subquestions(
        self,
        query: str,
        query_plan: Dict[str, Any]
    ) -> List[str]:
        """
        Decompose complex query into sub-questions.
        
        Args:
            query: Original query
            query_plan: Query plan
            
        Returns:
            List of sub-questions
        """
        if not self.llm_client:
            return []
        
        key_concepts = query_plan.get("key_concepts", [])
        
        prompt = f"""Decompose the following query into 2-4 specific sub-questions that need to be answered.

Query: {query}
Key concepts: {', '.join(key_concepts)}

Return only the sub-questions, one per line, without numbering or bullets."""

        try:
            response = self.llm_client.generate(prompt, max_tokens=200)
            sub_questions = [
                q.strip() for q in response.strip().split("\n")
                if q.strip() and not q.strip().startswith("#")
            ]
            return sub_questions[:4]  # Limit to 4 sub-questions
        except Exception as e:
            print(f"Warning: Sub-question decomposition failed: {e}")
            return []
    
    def _llm_relevance_judge(
        self,
        query: str,
        candidates: List[Candidate],
        sub_questions: Optional[List[str]] = None
    ) -> List[RefinedContext]:
        """
        Use LLM to judge relevance of candidates.
        
        Args:
            query: Original query
            candidates: Candidates to judge
            sub_questions: Optional sub-questions
            
        Returns:
            List of refined contexts with relevance scores
        """
        if not self.llm_client:
            return self._score_based_filter(candidates)
        
        refined_contexts = []
        
        # Batch process candidates for efficiency
        batch_size = 5
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            batch_results = self._judge_batch(query, batch, sub_questions)
            refined_contexts.extend(batch_results)
        
        # Sort by relevance score
        refined_contexts.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return refined_contexts
    
    def _judge_batch(
        self,
        query: str,
        candidates: List[Candidate],
        sub_questions: Optional[List[str]]
    ) -> List[RefinedContext]:
        """Judge a batch of candidates."""
        contexts_text = "\n\n".join([
            f"[Candidate {i+1}]\nContent: {c.content[:500]}\nScore: {c.score:.3f}"
            for i, c in enumerate(candidates)
        ])
        
        sub_questions_text = ""
        if sub_questions:
            sub_questions_text = f"\nSub-questions to consider:\n" + "\n".join(f"- {q}" for q in sub_questions)
        
        prompt = f"""Judge the relevance of each candidate document to the query.

Query: {query}
{sub_questions_text}

Candidates:
{contexts_text}

For each candidate, provide:
1. Relevance score (0.0 to 1.0)
2. Brief reason (one sentence)

Format as JSON array:
[
  {{"index": 1, "score": 0.85, "reason": "Directly explains..."}},
  ...
]"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=500)
            
            # Parse JSON response
            import json
            # Extract JSON from response (handle markdown code blocks)
            response_clean = response.strip()
            if "```json" in response_clean:
                response_clean = response_clean.split("```json")[1].split("```")[0].strip()
            elif "```" in response_clean:
                response_clean = response_clean.split("```")[1].split("```")[0].strip()
            
            judgments = json.loads(response_clean)
            
            refined = []
            for judgment in judgments:
                idx = judgment.get("index", 1) - 1
                if 0 <= idx < len(candidates):
                    candidate = candidates[idx]
                    refined.append(RefinedContext(
                        doc_id=candidate.doc_id,
                        content=candidate.content or "",
                        relevance_score=float(judgment.get("score", 0.5)),
                        reason=judgment.get("reason", "No reason provided"),
                        sub_question_match=judgment.get("sub_question") if "sub_question" in judgment else None,
                    ))
            
            return refined
        except Exception as e:
            print(f"Warning: LLM relevance judgment failed: {e}")
            # Fallback to score-based
            return self._score_based_filter(candidates)
    
    def _score_based_filter(self, candidates: List[Candidate]) -> List[RefinedContext]:
        """Fallback: filter based on retrieval scores."""
        return [
            RefinedContext(
                doc_id=candidate.doc_id,
                content=candidate.content or "",
                relevance_score=candidate.score,
                reason=f"Retrieved with score {candidate.score:.3f}",
            )
            for candidate in candidates
        ]
