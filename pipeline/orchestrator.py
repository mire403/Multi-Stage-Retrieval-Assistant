"""
Pipeline Orchestrator - Coordinates the multi-stage retrieval pipeline.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import yaml
from pathlib import Path

from query.analyzer import QueryAnalyzer, QueryPlan
from query.planner import QueryPlanner
from retrieval.stage1_broad import Stage1BroadRetriever, Candidate
from retrieval.stage2_refine import Stage2Refiner, RefinedContext
from synthesis.answer_generator import AnswerGenerator, Answer
from synthesis.citation_builder import CitationBuilder
from storage.vector_store import VectorStore
from storage.doc_store import DocumentStore, Document


class PipelineTrace(BaseModel):
    """Trace of pipeline execution."""
    query: str = Field(..., description="Original query")
    query_plan: Dict[str, Any] = Field(..., description="Query plan")
    execution_plan: Dict[str, Any] = Field(..., description="Execution plan")
    stage1_candidates: List[Dict[str, Any]] = Field(default_factory=list, description="Stage 1 candidates")
    stage2_contexts: List[Dict[str, Any]] = Field(default_factory=list, description="Stage 2 contexts")
    stage3_answer: Optional[Dict[str, Any]] = Field(default=None, description="Stage 3 answer")
    execution_time: Optional[float] = Field(default=None, description="Total execution time in seconds")


class LLMClient:
    """
    Simple LLM client wrapper.
    Can be extended to support different providers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM client.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-4-turbo-preview")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2000)
        
        self._init_client()
    
    def _init_client(self):
        """Initialize the LLM client based on provider."""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                import os
                api_key = os.getenv(self.config.get("api_key_env", "OPENAI_API_KEY"))
                if not api_key:
                    raise ValueError(f"API key not found in environment variable {self.config.get('api_key_env')}")
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("OpenAI package required. Install with: pip install openai")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate text using LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            return response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


class LayeredRetrieverPipeline:
    """
    Main pipeline orchestrator for multi-stage retrieval.
    
    Coordinates:
    1. Query Analysis
    2. Stage 1: Broad Retrieval
    3. Stage 2: Semantic Refinement
    4. Stage 3: Answer Synthesis
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to YAML configuration file
            config: Configuration dictionary (overrides config_path)
        """
        # Load configuration
        if config:
            self.config = config
        elif config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            # Default config
            default_config_path = Path(__file__).parent.parent / "config" / "retriever.yaml"
            if default_config_path.exists():
                with open(default_config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
            else:
                raise ValueError("No configuration provided")
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize all pipeline components."""
        # LLM Client
        llm_config = self.config.get("llm", {})
        self.llm_client = LLMClient(llm_config)
        
        # Storage
        storage_config = self.config.get("storage", {})
        vector_config = storage_config.get("vector_store", {})
        doc_config = storage_config.get("doc_store", {})
        
        self.vector_store = VectorStore(vector_config)
        self.doc_store = DocumentStore(doc_config)
        
        # Query Analysis
        query_config = self.config.get("query_analysis", {})
        self.query_analyzer = QueryAnalyzer(query_config)
        self.query_planner = QueryPlanner(self.config)
        
        # Retrieval
        stage1_config = self.config.get("stage1", {})
        stage2_config = self.config.get("stage2", {})
        
        self.stage1_retriever = Stage1BroadRetriever(
            self.vector_store,
            self.doc_store,
            stage1_config
        )
        
        from retrieval.ranker import Ranker
        ranker = Ranker(stage2_config)
        
        self.stage2_refiner = Stage2Refiner(
            self.llm_client,
            ranker,
            self.config
        )
        
        # Synthesis
        stage3_config = self.config.get("stage3", {})
        self.citation_builder = CitationBuilder(stage3_config)
        self.answer_generator = AnswerGenerator(
            self.llm_client,
            self.citation_builder,
            self.config
        )
        
        # Pipeline settings
        pipeline_config = self.config.get("pipeline", {})
        self.enable_dry_run = pipeline_config.get("enable_dry_run", True)
        self.dry_run_stage = pipeline_config.get("dry_run_stage")
        self.enable_tracing = pipeline_config.get("enable_tracing", True)
        self.trace_output_path = pipeline_config.get("trace_output_path", "./data/traces")
    
    def process(
        self,
        query: str,
        dry_run_stage: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the full pipeline.
        
        Args:
            query: User query
            dry_run_stage: Optional stage to stop at (stage1, stage2, stage3)
            
        Returns:
            Dictionary with answer and trace
        """
        import time
        start_time = time.time()
        
        trace = PipelineTrace(query=query)
        
        # Stage 0: Query Analysis
        query_plan = self.query_analyzer.analyze(query)
        execution_plan = self.query_planner.plan(query_plan)
        
        trace.query_plan = query_plan.dict()
        trace.execution_plan = execution_plan
        
        # Check for dry run
        dry_run = dry_run_stage or self.dry_run_stage
        
        # Stage 1: Broad Retrieval
        if not dry_run or dry_run == "stage1":
            candidates = self.stage1_retriever.retrieve(
                query,
                key_concepts=query_plan.key_concepts,
                constraints=query_plan.constraints,
                execution_plan=execution_plan.get("stage1", {})
            )
            trace.stage1_candidates = [c.dict() for c in candidates]
            
            if dry_run == "stage1":
                trace.execution_time = time.time() - start_time
                return {"trace": trace.dict(), "candidates": [c.dict() for c in candidates]}
        else:
            candidates = []
        
        # Stage 2: Semantic Refinement
        if not dry_run or dry_run in ["stage1", "stage2"]:
            contexts = self.stage2_refiner.refine(
                query,
                candidates,
                query_plan=query_plan.dict(),
                execution_plan=execution_plan.get("stage2", {})
            )
            trace.stage2_contexts = [c.dict() for c in contexts]
            
            if dry_run == "stage2":
                trace.execution_time = time.time() - start_time
                return {"trace": trace.dict(), "contexts": [c.dict() for c in contexts]}
        else:
            contexts = []
        
        # Stage 3: Answer Synthesis
        if not dry_run:
            answer = self.answer_generator.generate(
                query,
                contexts,
                query_plan=query_plan.dict(),
                execution_plan=execution_plan.get("stage3", {})
            )
            trace.stage3_answer = answer.dict()
        else:
            answer = None
        
        trace.execution_time = time.time() - start_time
        
        # Save trace if enabled
        if self.enable_tracing:
            self._save_trace(trace)
        
        return {
            "answer": answer.dict() if answer else None,
            "trace": trace.dict(),
        }
    
    def _save_trace(self, trace: PipelineTrace):
        """Save execution trace to file."""
        import json
        from datetime import datetime
        
        Path(self.trace_output_path).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_file = Path(self.trace_output_path) / f"trace_{timestamp}.json"
        
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(trace.dict(), f, indent=2, ensure_ascii=False)
