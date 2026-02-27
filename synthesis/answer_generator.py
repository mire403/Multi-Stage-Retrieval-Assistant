"""
Answer Generator - Generates final answers from refined contexts.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from retrieval.stage2_refine import RefinedContext
from synthesis.citation_builder import Citation


class Answer(BaseModel):
    """Final answer model."""
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Citations")
    confidence: float = Field(..., description="Confidence score (0-1)")
    reasoning_trace: Optional[str] = Field(default=None, description="Reasoning trace")
    used_contexts: List[str] = Field(default_factory=list, description="Document IDs used")


class AnswerGenerator:
    """
    Generates final answers from refined contexts.
    Enforces citation requirements and structured output.
    """
    
    def __init__(
        self,
        llm_client,
        citation_builder,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Answer Generator.
        
        Args:
            llm_client: LLM client
            citation_builder: CitationBuilder instance
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.citation_builder = citation_builder
        self.config = config or {}
        
        stage3_config = self.config.get("stage3", {})
        self.max_context_tokens = stage3_config.get("max_context_tokens", 4000)
        self.require_citations = stage3_config.get("require_citations", True)
        self.output_format = stage3_config.get("output_format", "structured")
        self.confidence_threshold = stage3_config.get("confidence_threshold", 0.6)
    
    def generate(
        self,
        query: str,
        contexts: List[RefinedContext],
        query_plan: Optional[Dict[str, Any]] = None,
        execution_plan: Optional[Dict[str, Any]] = None
    ) -> Answer:
        """
        Generate final answer from refined contexts.
        
        Args:
            query: Original user query
            contexts: Refined contexts from Stage 2
            query_plan: Query plan from analyzer
            execution_plan: Stage 3 execution plan
            
        Returns:
            Generated answer with citations
        """
        if execution_plan:
            # Override config with execution plan
            self.max_context_tokens = execution_plan.get(
                "max_context_tokens",
                self.max_context_tokens
            )
            self.require_citations = execution_plan.get(
                "require_citations",
                self.require_citations
            )
            self.output_format = execution_plan.get("output_format", self.output_format)
        
        if not contexts:
            return Answer(
                answer="I could not find sufficient information to answer your query.",
                confidence=0.0,
                reasoning_trace="No relevant contexts found in Stage 2 refinement.",
            )
        
        # Prepare context for LLM
        context_text = self._prepare_contexts(contexts)
        
        # Generate answer
        answer_text = self._generate_answer(query, context_text, query_plan)
        
        # Build citations
        citations = self.citation_builder.build_citations(contexts, answer_text)
        
        # Calculate confidence
        confidence = self._calculate_confidence(contexts, answer_text)
        
        # Extract reasoning trace if available
        reasoning_trace = self._extract_reasoning_trace(answer_text)
        
        # Clean answer (remove reasoning trace if embedded)
        clean_answer = self._clean_answer(answer_text)
        
        # Get used context IDs
        used_contexts = [ctx.doc_id for ctx in contexts]
        
        return Answer(
            answer=clean_answer,
            citations=citations,
            confidence=confidence,
            reasoning_trace=reasoning_trace,
            used_contexts=used_contexts,
        )
    
    def _prepare_contexts(self, contexts: List[RefinedContext]) -> str:
        """
        Prepare contexts for LLM input.
        
        Args:
            contexts: Refined contexts
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, context in enumerate(contexts, 1):
            context_parts.append(
                f"[Context {i} - Doc ID: {context.doc_id}, Relevance: {context.relevance_score:.2f}]\n"
                f"{context.content}\n"
            )
        
        context_text = "\n".join(context_parts)
        
        # Truncate if too long (rough token estimation: 1 token ≈ 4 chars)
        max_chars = self.max_context_tokens * 4
        if len(context_text) > max_chars:
            # Truncate from the end, but try to keep complete contexts
            truncated = ""
            for part in context_parts:
                if len(truncated) + len(part) <= max_chars:
                    truncated += part
                else:
                    break
            context_text = truncated + "\n[Additional contexts truncated...]"
        
        return context_text
    
    def _generate_answer(
        self,
        query: str,
        context_text: str,
        query_plan: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate answer using LLM.
        
        Args:
            query: User query
            context_text: Prepared context text
            query_plan: Query plan
            
        Returns:
            Generated answer
        """
        intent = query_plan.get("intent", "exploratory") if query_plan else "exploratory"
        
        # Build prompt based on intent
        if intent == "comparative":
            instruction = "Compare the information from the contexts and provide a structured comparison."
        elif intent == "procedural":
            instruction = "Provide step-by-step instructions based on the contexts."
        elif intent == "factual":
            instruction = "Provide a direct, factual answer based on the contexts."
        else:
            instruction = "Synthesize information from the contexts to answer the query."
        
        prompt = f"""You are a precise information retrieval assistant. Answer the query using ONLY the provided contexts.

{instruction}

Query: {query}

Contexts:
{context_text}

Instructions:
1. Answer the query using information from the contexts above.
2. If the contexts do not contain sufficient information, state that clearly.
3. Cite specific contexts when making claims (use [Context N] format).
4. Be precise and avoid speculation.
5. If the query requires information not in the contexts, acknowledge this limitation.

Answer:"""

        try:
            answer = self.llm_client.generate(prompt, max_tokens=2000)
            return answer.strip()
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def _calculate_confidence(
        self,
        contexts: List[RefinedContext],
        answer: str
    ) -> float:
        """
        Calculate confidence score for the answer.
        
        Args:
            contexts: Refined contexts
            answer: Generated answer
            
        Returns:
            Confidence score (0-1)
        """
        if not contexts:
            return 0.0
        
        # Base confidence on average relevance score
        avg_relevance = sum(ctx.relevance_score for ctx in contexts) / len(contexts)
        
        # Boost if multiple high-quality contexts
        context_count_factor = min(len(contexts) / 5.0, 1.0)
        
        # Penalize if answer is too short (might indicate insufficient info)
        answer_length_factor = min(len(answer) / 200.0, 1.0)
        
        confidence = (
            avg_relevance * 0.5 +
            context_count_factor * 0.2 +
            answer_length_factor * 0.3
        )
        
        return min(confidence, 1.0)
    
    def _extract_reasoning_trace(self, answer: str) -> Optional[str]:
        """
        Extract reasoning trace from answer if present.
        
        Args:
            answer: Generated answer
            
        Returns:
            Reasoning trace or None
        """
        # Look for reasoning sections (if LLM includes them)
        if "Reasoning:" in answer or "Trace:" in answer:
            # Simple extraction - can be enhanced
            parts = answer.split("Reasoning:")
            if len(parts) > 1:
                return parts[1].strip()
        
        return None
    
    def _clean_answer(self, answer: str) -> str:
        """
        Clean answer text (remove embedded traces, etc.).
        
        Args:
            answer: Raw answer
            
        Returns:
            Cleaned answer
        """
        # Remove reasoning sections if they're separate
        if "Reasoning:" in answer:
            answer = answer.split("Reasoning:")[0].strip()
        
        return answer
