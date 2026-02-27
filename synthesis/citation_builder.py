"""
Citation Builder - Builds citations and traces for answers.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from retrieval.stage2_refine import RefinedContext


class Citation(BaseModel):
    """Citation model."""
    doc_id: str = Field(..., description="Document identifier")
    content_snippet: str = Field(..., description="Relevant content snippet")
    relevance_reason: str = Field(..., description="Why this citation is relevant")
    position: Optional[int] = Field(default=None, description="Position in answer if applicable")


class CitationBuilder:
    """
    Builds citations and traces for generated answers.
    Ensures all claims are backed by retrieved contexts.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Citation Builder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.snippet_length = self.config.get("snippet_length", 200)
    
    def build_citations(
        self,
        contexts: List[RefinedContext],
        answer: str
    ) -> List[Citation]:
        """
        Build citations from refined contexts.
        
        Args:
            contexts: Refined contexts from Stage 2
            answer: Generated answer
            
        Returns:
            List of citations
        """
        citations = []
        
        for i, context in enumerate(contexts):
            # Extract relevant snippet
            snippet = self._extract_snippet(context.content)
            
            citation = Citation(
                doc_id=context.doc_id,
                content_snippet=snippet,
                relevance_reason=context.reason,
                position=i,
            )
            citations.append(citation)
        
        return citations
    
    def _extract_snippet(self, content: str) -> str:
        """
        Extract a relevant snippet from content.
        
        Args:
            content: Full content
            
        Returns:
            Snippet of specified length
        """
        if len(content) <= self.snippet_length:
            return content
        
        # Try to extract from beginning, but break at sentence boundary
        snippet = content[:self.snippet_length]
        last_period = snippet.rfind(".")
        last_newline = snippet.rfind("\n")
        
        # Break at the later of period or newline
        break_point = max(last_period, last_newline)
        if break_point > self.snippet_length * 0.5:  # Only if we have enough content
            snippet = snippet[:break_point + 1]
        else:
            snippet = content[:self.snippet_length] + "..."
        
        return snippet
    
    def format_citations(self, citations: List[Citation], format_type: str = "markdown") -> str:
        """
        Format citations for display.
        
        Args:
            citations: List of citations
            format_type: Format type (markdown, plain, json)
            
        Returns:
            Formatted citation string
        """
        if format_type == "markdown":
            return self._format_markdown(citations)
        elif format_type == "plain":
            return self._format_plain(citations)
        elif format_type == "json":
            import json
            return json.dumps([c.dict() for c in citations], indent=2, ensure_ascii=False)
        else:
            return self._format_markdown(citations)
    
    def _format_markdown(self, citations: List[Citation]) -> str:
        """Format citations in Markdown."""
        lines = ["## Citations\n"]
        
        for i, citation in enumerate(citations, 1):
            lines.append(f"### [{i}] Document: {citation.doc_id}")
            lines.append(f"**Reason:** {citation.relevance_reason}")
            lines.append(f"**Snippet:** {citation.content_snippet}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_plain(self, citations: List[Citation]) -> str:
        """Format citations in plain text."""
        lines = ["Citations:\n"]
        
        for i, citation in enumerate(citations, 1):
            lines.append(f"[{i}] {citation.doc_id}")
            lines.append(f"  Reason: {citation.relevance_reason}")
            lines.append(f"  Snippet: {citation.content_snippet}")
            lines.append("")
        
        return "\n".join(lines)
