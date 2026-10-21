from src.pipelines.graph import build_graph

def test_pipeline_execution():
    app = build_graph()
    query = "Which beverage brands under Coca-Cola saw distribution gains in Modern Trade across APAC in Q3?"
    initial_state = {
        "query": query,
        "extracted_entities": [],
        "intent": "point_lookup",
        "vector_results": [],
        "graph_results": [],
        "fused_results": [],
        "final_response": None,
        "is_grounded": False,
        "retry_count": 0
    }

    result = app.invoke(initial_state)
    assert result is not None
    assert result["is_grounded"] is True
    assert result["final_response"] is not None
    assert result["final_response"].confidence_score >= 0.8
    assert len(result["final_response"].sources) > 0
