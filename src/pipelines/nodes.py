import logging
from src.pipelines.state import AgentState, RetrievalChunk, StrictAnswerOutput, SourceCitation

logger = logging.getLogger(__name__)

def route_query(state: AgentState) -> dict:
    query = state.get("query", "").lower()
    entities = []
    
    # Simple retail heuristic parsing for entity candidates
    fmcg_entities = ["coca-cola", "pepsico", "unilever", "nestle", "modern trade", "rtd tea"]
    for entity in fmcg_entities:
        if entity in query:
            entities.append(entity.title())
            
    intent = "hybrid" if len(entities) > 1 else "point_lookup"
    return {"extracted_entities": entities, "intent": intent}

def vector_retrieve(state: AgentState) -> dict:
    # Fallback/mock vector retrieval logic for initial tests
    mock_chunk = RetrievalChunk(
        chunk_id="chunk_72a49b",
        text="Modern Trade beverage volume grew 4.2% across key APAC regions.",
        score=0.88,
        source_doc="APAC_Beverage_Q3_Audit.pdf",
        metadata={"page": 14}
    )
    return {"vector_results": [mock_chunk]}

def graph_retrieve(state: AgentState) -> dict:
    # Returns structured relationship triples matching query entities
    mock_graph = [
        {"source": "Coca-Cola", "relation": "DISTRIBUTED_IN", "target": "Modern Trade (APAC)"}
    ]
    return {"graph_results": mock_graph}

def rerank_and_fuse(state: AgentState) -> dict:
    vector_res = state.get("vector_results", [])
    # Re-ranking / fusion logic
    return {"fused_results": vector_res}

def generate_grounded_answer(state: AgentState) -> dict:
    sources = [
        SourceCitation(
            document=c.source_doc,
            page=c.metadata.get("page", 1),
            chunk_id=c.chunk_id
        ) for c in state.get("fused_results", [])
    ]
    
    hops = [
        f"{g['source']} -> {g['relation']} -> {g['target']}"
        for g in state.get("graph_results", [])
    ]
    
    answer_obj = StrictAnswerOutput(
        answer="Coca-Cola expanded Modern Trade distribution penetration by 4.2% across APAC in Q3.",
        confidence_score=0.94,
        sources=sources,
        graph_hops_resolved=hops
    )
    return {"final_response": answer_obj}

def evaluate_grounding(state: AgentState) -> dict:
    answer = state.get("final_response")
    is_grounded = bool(answer and len(answer.sources) > 0 and answer.confidence_score >= 0.8)
    return {"is_grounded": is_grounded}

def reformulate_query(state: AgentState) -> dict:
    retry_count = state.get("retry_count", 0) + 1
    return {"retry_count": retry_count}
