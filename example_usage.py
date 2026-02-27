"""
Example usage of LayeredRetriever pipeline.

This script demonstrates how to:
1. Initialize the pipeline
2. Add documents to the system
3. Process queries
4. View results and traces
"""

from pipeline.orchestrator import LayeredRetrieverPipeline
from storage.vector_store import VectorStore
from storage.doc_store import DocumentStore, Document
from pathlib import Path
import yaml


def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "config" / "retriever.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_sample_documents():
    """Set up sample documents for testing."""
    config = load_config()
    
    # Initialize stores
    storage_config = config.get("storage", {})
    vector_store = VectorStore(storage_config.get("vector_store", {}))
    doc_store = DocumentStore(storage_config.get("doc_store", {}))
    
    # Sample documents about multi-stage retrieval
    sample_docs = [
        {
            "doc_id": "doc1",
            "content": """
            Multi-stage retrieval is a technique used in information retrieval systems to improve 
            both recall and precision. The first stage typically uses fast, approximate methods 
            to retrieve a large candidate set, while subsequent stages apply more expensive but 
            accurate methods to refine the results.
            """,
            "metadata": {"topic": "retrieval", "type": "concept"}
        },
        {
            "doc_id": "doc2",
            "content": """
            Stage 1 retrieval in multi-stage systems focuses on high recall. Common approaches 
            include vector similarity search, keyword matching, and metadata filtering. The goal 
            is to not miss potentially relevant documents, even if some irrelevant ones are included.
            """,
            "metadata": {"topic": "retrieval", "type": "method", "stage": 1}
        },
        {
            "doc_id": "doc3",
            "content": """
            Stage 2 refinement uses more sophisticated methods to filter and re-rank candidates 
            from Stage 1. Techniques include cross-encoder models, LLM-based relevance judgment, 
            and sub-question decomposition. The goal is high precision while maintaining recall.
            """,
            "metadata": {"topic": "retrieval", "type": "method", "stage": 2}
        },
        {
            "doc_id": "doc4",
            "content": """
            Transformer architecture is a deep learning model architecture introduced in the paper 
            "Attention Is All You Need". It relies entirely on self-attention mechanisms and has 
            become the foundation for many modern NLP models including BERT, GPT, and T5.
            """,
            "metadata": {"topic": "transformer", "type": "concept"}
        },
        {
            "doc_id": "doc5",
            "content": """
            BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based 
            model that uses bidirectional context. Unlike GPT which is autoregressive, BERT uses 
            masked language modeling during pre-training.
            """,
            "metadata": {"topic": "bert", "type": "model"}
        },
    ]
    
    # Add to vector store
    vector_store.add_documents(sample_docs)
    
    # Add to document store
    documents = [Document(**doc) for doc in sample_docs]
    doc_store.add_documents(documents)
    
    print(f"✓ Added {len(sample_docs)} sample documents")
    return vector_store, doc_store


def example_basic_query():
    """Example 1: Basic query processing."""
    print("\n" + "="*60)
    print("Example 1: Basic Query Processing")
    print("="*60)
    
    # Initialize pipeline
    config_path = Path(__file__).parent / "config" / "retriever.yaml"
    pipeline = LayeredRetrieverPipeline(config_path=str(config_path))
    
    # Process query
    query = "What is multi-stage retrieval?"
    print(f"\nQuery: {query}\n")
    
    result = pipeline.process(query)
    
    # Display results
    if result["answer"]:
        answer = result["answer"]
        print("Answer:")
        print("-" * 60)
        print(answer["answer"])
        print("-" * 60)
        print(f"\nConfidence: {answer['confidence']:.2f}")
        print(f"Citations: {len(answer['citations'])}")
        print(f"\nUsed Contexts: {len(answer['used_contexts'])}")
    
    # Display trace summary
    trace = result["trace"]
    print(f"\nTrace Summary:")
    print(f"  - Stage 1 Candidates: {len(trace['stage1_candidates'])}")
    print(f"  - Stage 2 Contexts: {len(trace['stage2_contexts'])}")
    print(f"  - Execution Time: {trace['execution_time']:.2f}s")


def example_comparative_query():
    """Example 2: Comparative query."""
    print("\n" + "="*60)
    print("Example 2: Comparative Query")
    print("="*60)
    
    config_path = Path(__file__).parent / "config" / "retriever.yaml"
    pipeline = LayeredRetrieverPipeline(config_path=str(config_path))
    
    query = "Compare transformer and BERT architectures"
    print(f"\nQuery: {query}\n")
    
    result = pipeline.process(query)
    
    if result["answer"]:
        answer = result["answer"]
        print("Answer:")
        print("-" * 60)
        print(answer["answer"])
        print("-" * 60)
        
        # Show query plan
        trace = result["trace"]
        query_plan = trace["query_plan"]
        print(f"\nQuery Plan:")
        print(f"  - Intent: {query_plan['intent']}")
        print(f"  - Key Concepts: {', '.join(query_plan['key_concepts'])}")
        print(f"  - Complexity: {query_plan['complexity_score']:.2f}")


def example_dry_run():
    """Example 3: Dry run to inspect intermediate stages."""
    print("\n" + "="*60)
    print("Example 3: Dry Run (Stage 1 Only)")
    print("="*60)
    
    config_path = Path(__file__).parent / "config" / "retriever.yaml"
    pipeline = LayeredRetrieverPipeline(config_path=str(config_path))
    
    query = "How does multi-stage retrieval work?"
    print(f"\nQuery: {query}\n")
    
    # Dry run to Stage 1
    result = pipeline.process(query, dry_run_stage="stage1")
    
    print("Stage 1 Candidates (Top 5):")
    print("-" * 60)
    candidates = result.get("candidates", [])
    for i, candidate in enumerate(candidates[:5], 1):
        print(f"{i}. Doc ID: {candidate['doc_id']}")
        print(f"   Score: {candidate['score']:.3f}")
        print(f"   Source: {candidate['source']}")
        print()


def example_trace_inspection():
    """Example 4: Inspect full execution trace."""
    print("\n" + "="*60)
    print("Example 4: Full Trace Inspection")
    print("="*60)
    
    config_path = Path(__file__).parent / "config" / "retriever.yaml"
    pipeline = LayeredRetrieverPipeline(config_path=str(config_path))
    
    query = "What are the stages in multi-stage retrieval?"
    print(f"\nQuery: {query}\n")
    
    result = pipeline.process(query)
    trace = result["trace"]
    
    print("Full Execution Trace:")
    print("-" * 60)
    print(f"Query Plan:")
    print(f"  Intent: {trace['query_plan']['intent']}")
    print(f"  Needs Multi-hop: {trace['query_plan']['needs_multi_hop']}")
    print(f"  Key Concepts: {trace['query_plan']['key_concepts']}")
    
    print(f"\nStage 1 (Broad Retrieval):")
    print(f"  Candidates: {len(trace['stage1_candidates'])}")
    if trace['stage1_candidates']:
        top_candidate = trace['stage1_candidates'][0]
        print(f"  Top Candidate: {top_candidate['doc_id']} (score: {top_candidate['score']:.3f})")
    
    print(f"\nStage 2 (Semantic Refinement):")
    print(f"  Contexts: {len(trace['stage2_contexts'])}")
    if trace['stage2_contexts']:
        top_context = trace['stage2_contexts'][0]
        print(f"  Top Context: {top_context['doc_id']} (relevance: {top_context['relevance_score']:.3f})")
        print(f"  Reason: {top_context['reason']}")
    
    print(f"\nExecution Time: {trace['execution_time']:.2f}s")


if __name__ == "__main__":
    print("LayeredRetriever - Example Usage")
    print("="*60)
    
    # Setup sample documents
    try:
        setup_sample_documents()
    except Exception as e:
        print(f"Warning: Could not setup sample documents: {e}")
        print("Make sure you have documents in your stores.")
    
    # Run examples
    try:
        example_basic_query()
        example_comparative_query()
        example_dry_run()
        example_trace_inspection()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. You have set OPENAI_API_KEY environment variable")
        print("2. You have installed all dependencies: pip install -r requirements.txt")
        print("3. You have documents in your vector store and document store")
