# Agentic RAG Knowledge Platform
Last Updated: Sun Sep 6, 2026 | Maintained by Alexander Ioffrida

## Overview
*Building a hybrid-search RAG service that routes queries across retrieval, web fallback, and a reranker, with citations and an offline eval harness that scores faithfulness.*

This production-grade RAG service will combine dense vector search (Qdrant) with BM25 sparse retrieval, then route low-confidence queries to a live web fallback before a BGE cross-encoder reranker selects the final context passages. A LangGraph agent orchestrates the routing logic, the FastAPI layer serves grounded layers with inline citations, and a Ragas eval harness runs offline to score faithfulness, context precision, and answer relevancy on a held-out question set. This stack covers the full lifecycle that teams actually deploy: ingestion, hybrid retrieval, agentic routing, reranking, citation formatting, and quantitative evaluation.

## The Stack
LangGraph • pgvector • Qdrant • FastAPI • BGE reranker • Ragas • sentence-transformers • OpenRouter