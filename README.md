# MarketNexus RAG: Enterprise FMCG & Retail Intelligence Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Neo4j](https://img.shields.io/badge/GraphDB-Neo4j%205.x-blue.svg)](https://neo4j.com/)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red.svg)](https://qdrant.tech/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

MarketNexus RAG is an end-to-end, enterprise-grade hybrid retrieval-augmented generation engine tailored for Fast-Moving Consumer Goods (FMCG) and retail market measurement analytics. It bridges the gap between unstructured market audit reports and structured, multi-hop category taxonomies by fusing **Knowledge Graphs (Neo4j)** with **Hybrid Vector Search (Qdrant)** under a cyclic **LangGraph** orchestration loop.

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
