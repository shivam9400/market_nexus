from typing import List, Optional, Literal, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class SourceCitation(BaseModel):
    document: str
    page: int
    chunk_id: str

class StrictAnswerOutput(BaseModel):
    answer: str = Field(description="Synthesized market intelligence narrative.")
    confidence_score: float = Field(ge=0.0, le=1.0)
    sources: List[SourceCitation] = Field(default_factory=list)
    graph_hops_resolved: List[str] = Field(default_factory=list)

class RetrievalChunk(BaseModel):
    chunk_id: str
    text: str
    score: float = 0.0
    source_doc: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentState(TypedDict):
    query: str
    extracted_entities: List[str]
    intent: Literal["point_lookup", "relational_multi_hop", "hybrid"]
    vector_results: List[RetrievalChunk]
    graph_results: List[Dict[str, Any]]
    fused_results: List[RetrievalChunk]
    final_response: Optional[StrictAnswerOutput]
    is_grounded: bool
    retry_count: int
