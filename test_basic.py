"""
Basic test script to verify LayeredRetriever installation and basic functionality.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from query.analyzer import QueryAnalyzer, QueryPlan
        from query.planner import QueryPlanner
        print("✓ Query modules imported")
    except Exception as e:
        print(f"✗ Query modules failed: {e}")
        return False
    
    try:
        from storage.vector_store import VectorStore
        from storage.doc_store import DocumentStore, Document
        print("✓ Storage modules imported")
    except Exception as e:
        print(f"✗ Storage modules failed: {e}")
        return False
    
    try:
        from retrieval.stage1_broad import Stage1BroadRetriever
        from retrieval.stage2_refine import Stage2Refiner
        from retrieval.ranker import Ranker
        print("✓ Retrieval modules imported")
    except Exception as e:
        print(f"✗ Retrieval modules failed: {e}")
        return False
    
    try:
        from synthesis.answer_generator import AnswerGenerator
        from synthesis.citation_builder import CitationBuilder
        print("✓ Synthesis modules imported")
    except Exception as e:
        print(f"✗ Synthesis modules failed: {e}")
        return False
    
    try:
        from pipeline.orchestrator import LayeredRetrieverPipeline
        print("✓ Pipeline module imported")
    except Exception as e:
        print(f"✗ Pipeline module failed: {e}")
        return False
    
    return True


def test_query_analyzer():
    """Test query analyzer."""
    print("\nTesting Query Analyzer...")
    
    try:
        from query.analyzer import QueryAnalyzer
        
        analyzer = QueryAnalyzer()
        
        # Test factual query
        plan = analyzer.analyze("What is multi-stage retrieval?")
        assert plan.intent.value == "factual" or plan.intent.value == "exploratory"
        print(f"✓ Factual query analyzed: intent={plan.intent.value}")
        
        # Test comparative query
        plan = analyzer.analyze("Compare transformer and BERT")
        assert plan.intent.value == "comparative"
        print(f"✓ Comparative query analyzed: intent={plan.intent.value}")
        
        # Test procedural query
        plan = analyzer.analyze("How to implement multi-stage retrieval?")
        assert plan.intent.value == "procedural"
        print(f"✓ Procedural query analyzed: intent={plan.intent.value}")
        
        return True
    except Exception as e:
        print(f"✗ Query analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_store():
    """Test document store."""
    print("\nTesting Document Store...")
    
    try:
        from storage.doc_store import DocumentStore, Document
        
        store = DocumentStore({"type": "memory"})
        
        # Add a document
        doc = Document(
            doc_id="test1",
            content="This is a test document about multi-stage retrieval.",
            metadata={"topic": "retrieval"}
        )
        store.add_document(doc)
        
        # Retrieve it
        retrieved = store.get_document("test1")
        assert retrieved is not None
        assert retrieved.content == doc.content
        print("✓ Document store: add and retrieve")
        
        # Search by metadata
        results = store.search_by_metadata({"topic": "retrieval"})
        assert len(results) > 0
        print("✓ Document store: metadata search")
        
        return True
    except Exception as e:
        print(f"✗ Document store test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting Configuration Loading...")
    
    try:
        import yaml
        from pathlib import Path
        
        config_path = Path(__file__).parent / "config" / "retriever.yaml"
        
        if not config_path.exists():
            print("⚠ Config file not found, skipping test")
            return True
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        assert "llm" in config
        assert "stage1" in config
        assert "stage2" in config
        assert "stage3" in config
        print("✓ Configuration loaded successfully")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("LayeredRetriever - Basic Tests")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Query Analyzer", test_query_analyzer),
        ("Document Store", test_document_store),
        ("Configuration", test_config_loading),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
