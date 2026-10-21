from langgraph.graph import StateGraph, END
from src.pipelines.state import AgentState
from src.pipelines.nodes import (
    route_query,
    vector_retrieve,
    graph_retrieve,
    rerank_and_fuse,
    generate_grounded_answer,
    evaluate_grounding,
    reformulate_query,
)

def should_retry(state: AgentState):
    if state.get("is_grounded", False):
        return END
    if state.get("retry_count", 0) >= 2:
        return END
    return "reformulate_query"

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("route_query", route_query)
    workflow.add_node("vector_retrieve", vector_retrieve)
    workflow.add_node("graph_retrieve", graph_retrieve)
    workflow.add_node("rerank_and_fuse", rerank_and_fuse)
    workflow.add_node("generate_grounded_answer", generate_grounded_answer)
    workflow.add_node("evaluate_grounding", evaluate_grounding)
    workflow.add_node("reformulate_query", reformulate_query)

    workflow.set_entry_point("route_query")

    workflow.add_edge("route_query", "vector_retrieve")
    workflow.add_edge("route_query", "graph_retrieve")
    workflow.add_edge("vector_retrieve", "rerank_and_fuse")
    workflow.add_edge("graph_retrieve", "rerank_and_fuse")
    workflow.add_edge("rerank_and_fuse", "generate_grounded_answer")
    workflow.add_edge("generate_grounded_answer", "evaluate_grounding")

    workflow.add_conditional_edges(
        "evaluate_grounding",
        should_retry,
        {
            END: END,
            "reformulate_query": "reformulate_query"
        }
    )
    workflow.add_edge("reformulate_query", "route_query")

    return workflow.compile()
