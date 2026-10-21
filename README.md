# MarketNexus RAG: Enterprise FMCG & Retail Intelligence Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Neo4j](https://img.shields.io/badge/GraphDB-Neo4j%205.x-blue.svg)](https://neo4j.com/)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red.svg)](https://qdrant.tech/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

MarketNexus RAG is an end-to-end, enterprise-grade hybrid retrieval-augmented generation engine tailored for Fast-Moving Consumer Goods (FMCG) and retail measurement analytics. It bridges the gap between unstructured market audit reports and structured, multi-hop category taxonomies by fusing **Knowledge Graphs (Neo4j)** with **Hybrid Vector Search (Qdrant)** under a cyclic **LangGraph** orchestration loop.

---

## Key Features

- **Multi-Modal Document Parsing:** Integrates `Docling` to extract embedded tables, balance sheets, and store audit metrics directly into structured Markdown without losing tabular alignment.
- **Hierarchical Knowledge Graph:** Maps retail domain ontologies (`Manufacturer` $\rightarrow$ `Brand` $\rightarrow$ `SKU`, `Channel`, and `Retailer`) to answer multi-hop, relational competitor queries.
- **Hybrid Retrieval & Fusion:** Combines dense semantic vectors with sparse BM25 tokens, followed by a reciprocal rank fusion (RRF) and Cross-Encoder reranking pass via `FlashRank`.
- **Cyclic Agentic State Machine:** Built on `LangGraph` with self-corrective query reformulation, dynamic routing (Vector vs. Graph vs. Hybrid), and automated hallucination/grounding checks.
- **Contract-Enforced JSON Output:** Powered by `Instructor` and `Pydantic v2` to guarantee typed responses, metric precision, and verifiable chunk-level citations.
- **Automated CI/CD Evals & Tracing:** Out-of-the-box support for `Ragas` test suites and continuous `Langfuse` telemetry for latency, token economics, and retrieval relevance.

---

## Architecture Overview

```text
User Query
    │
    ▼
[ route_query ] (Extract Entities, Detect Intent: Vector / Graph / Hybrid)
    │
    ├───► [ graph_retrieve ]  (Cypher Query Execution over Neo4j)
    │            │
    └───► [ vector_retrieve ] (Dense + BM25 Hybrid Search over Qdrant)
                 │
                 ▼
        [ rerank_and_fuse ]   (Reciprocal Rank Fusion & Cross-Encoder)
                 │
                 ▼
    [ generate_grounded_answer ] (Instructor-enforced Pydantic Output)
                 │
                 ▼
       [ evaluate_grounding ] ──(Hallucination / Low Confidence)──► [ reformulate_query ]
                 │                                                          │
                 │ (Passed / Confidence OK)                                 └──► (Retry Loop)
                 ▼
             Final Output
```

---

## Directory Structure

```text
marketnexus-rag/
├── config/
│   ├── __init__.py
│   └── settings.py              # Pydantic BaseSettings for env management
├── data/
│   ├── sample_reports/          # FMCG market share and category audit decks
│   └── synthetic_eval_set.json  # Benchmark questions for RAGAS evaluation
├── docker/
│   ├── docker-compose.yml       # Neo4j and Qdrant container configurations
│   └── Dockerfile               # Production service container
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parser.py            # Docling parsing & markdown table conversion
│   │   ├── graph_builder.py     # Entity-relation extraction & Neo4j loading
│   │   └── vector_indexer.py    # Dense + sparse vector embeddings into Qdrant
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── state.py             # AgentState TypedDict and Pydantic output schemas
│   │   ├── nodes.py             # Individual LangGraph execution nodes
│   │   ├── edges.py             # Conditional routing and validation edges
│   │   └── graph.py             # Compiled StateGraph workflow engine
│   ├── utils/
│   │   ├── __init__.py
│   │   └── reranker.py          # Cross-encoder & RRF ranking utilities
│   └── evaluate.py              # Ragas benchmark execution harness
├── .env.example
├── main.py                      # CLI & API execution entrypoint
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- API Keys: OpenAI API key (or local Ollama/vLLM instance) and optional Langfuse keys

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/your-org/marketnexus-rag.git
cd marketnexus-rag

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Local Data Infrastructure

Start the local instances for **Neo4j** (Knowledge Graph) and **Qdrant** (Vector Engine):

```bash
docker-compose -f docker/docker-compose.yml up -d
```

- **Neo4j Browser:** `http://localhost:7474` (Default user: `neo4j`, password: `password123`)
- **Qdrant Dashboard:** `http://localhost:6333/dashboard`

### 3. Configure Secrets

Copy `.env.example` to `.env` and supply your credentials:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY="sk-..."
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="password123"
QDRANT_URL="http://localhost:6333"
QDRANT_COLLECTION="fmcg_intelligence"
LANGFUSE_PUBLIC_KEY="pk-..."
LANGFUSE_SECRET_KEY="sk-..."
LANGFUSE_HOST="https://cloud.langfuse.com"
```

---

## Usage

### Ingesting Market Intelligence Reports

Run the ingestion pipeline to parse unstructured reports, extract retail ontologies into Neo4j, and embed hybrid chunks into Qdrant:

```bash
python -m src.ingestion.parser --input-dir data/sample_reports/
```

### Querying the System

Execute an analytical multi-hop query via the main driver:

```bash
python main.py --query "Which competing beverage brands under Pepsico and Coca-Cola saw distribution gains in Modern Trade across APAC in Q3?"
```

Sample output:

```json
{
  "answer": "During Q3, Coca-Cola's Ready-to-Drink tea category expanded Modern Trade penetration by 4.2% across key APAC markets, offsetting a 1.8% contraction in PepsiCo's ambient juice portfolio...",
  "confidence_score": 0.94,
  "sources": [
    {
      "document": "APAC_Beverage_Q3_Audit.pdf",
      "page": 14,
      "chunk_id": "chunk_72a49b"
    }
  ],
  "graph_hops_resolved": [
    "Coca-Cola -> MANUFACTURES -> RTD Tea -> DISTRIBUTED_IN -> Modern Trade (APAC)"
  ]
}
```

---

## Running Automated Evaluations

Run the automated evaluation benchmark harness using `Ragas` to measure Answer Relevance, Context Precision, and Faithfulness:

```bash
python -m src.evaluate --dataset data/synthetic_eval_set.json
```

Results are logged directly to your terminal and synced to your `Langfuse` dashboard for trace inspection.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
