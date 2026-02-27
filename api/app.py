"""
API Layer - FastAPI application for LayeredRetriever.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn
from pathlib import Path

from pipeline.orchestrator import LayeredRetrieverPipeline


# Request/Response Models
class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., description="User query")
    dry_run_stage: Optional[str] = Field(
        default=None,
        description="Optional stage to stop at (stage1, stage2, stage3)"
    )


class QueryResponse(BaseModel):
    """Query response model."""
    answer: Optional[Dict[str, Any]] = Field(default=None, description="Generated answer")
    trace: Dict[str, Any] = Field(..., description="Execution trace")
    success: bool = Field(default=True, description="Whether the request succeeded")


# Initialize FastAPI app
app = FastAPI(
    title="LayeredRetriever API",
    description="Multi-Stage Retrieval Assistant API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[LayeredRetrieverPipeline] = None


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline
    try:
        config_path = Path(__file__).parent.parent / "config" / "retriever.yaml"
        pipeline = LayeredRetrieverPipeline(config_path=str(config_path))
    except Exception as e:
        print(f"Warning: Could not initialize pipeline: {e}")
        print("Pipeline will be initialized on first request.")


def get_pipeline() -> LayeredRetrieverPipeline:
    """Get or initialize pipeline."""
    global pipeline
    if pipeline is None:
        config_path = Path(__file__).parent.parent / "config" / "retriever.yaml"
        pipeline = LayeredRetrieverPipeline(config_path=str(config_path))
    return pipeline


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LayeredRetriever API",
        "version": "1.0.0",
        "description": "Multi-Stage Retrieval Assistant"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a query through the multi-stage retrieval pipeline.
    
    Args:
        request: Query request
        
    Returns:
        Query response with answer and trace
    """
    try:
        pipeline_instance = get_pipeline()
        result = pipeline_instance.process(
            query=request.query,
            dry_run_stage=request.dry_run_stage
        )
        
        return QueryResponse(
            answer=result.get("answer"),
            trace=result.get("trace", {}),
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trace/{trace_id}")
async def get_trace(trace_id: str):
    """
    Get a saved trace by ID.
    
    Args:
        trace_id: Trace identifier
        
    Returns:
        Trace data
    """
    # Implementation would load trace from file system
    return {"message": "Trace retrieval not yet implemented", "trace_id": trace_id}


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
