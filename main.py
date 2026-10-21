import argparse
import json
from src.pipelines.graph import build_graph

def run_query(query: str):
    app = build_graph()
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
    
    final_output = app.invoke(initial_state)
    response = final_output.get("final_response")
    if response:
        print(json.dumps(response.model_dump(), indent=2))
    else:
        print(json.dumps({"error": "Failed to generate grounded answer"}, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarketNexus RAG Engine")
    parser.add_argument("--query", type=str, required=True, help="Input market intelligence question")
    args = parser.parse_args()
    run_query(args.query)
