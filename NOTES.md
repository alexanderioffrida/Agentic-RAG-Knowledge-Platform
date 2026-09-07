# NOTES

Personal notes on urgent items to be fixed / implemented.

## URGENT
**Candidate Pooling.** HF docs, transformers docs, and web are three separate tools, and the candidate pool structurally cannot exist. The LLM commits to a source *before* any scoring happens, and the reranker would only ever see candidates from whichever tool's pool got picked. I'd be reranking within a source rather than across sources.
**Parallel Retrieval / RAG Accuracy.** Query both the BM25 sparse index (once it's made) and the dense vector index at the same time to fetch top candidate list. Then apply RRF (No Score Normalization Needed). Apply reranking with BGE cross-encoder reranker.

## Primary

**FUTURE: FastAPI Layer.** `MemorySaver` is in-process only, so history dies with the REPL. when i get to the FastAPI layer i'll want a persistent checkpointer and a `thread_id` per user rather than a constant.

## Secondary

**REPL #2.** with a single hardcoded `thread_id`, history grows without bound, including every 700-token retrieved chunk, so a long session will creep toward the context limit and rising per-turn cost.

- `.load()[:50]` still materializes the whole dataset before slicing.
- `route` is still a manual reimplementation of `tools_condition`.
- No env-var validation.
- No type hints on `preprocess_dataset`/`create_retriever`.
- Hybrid retrieval, the BGE ranker, and citations are all still absent.
- I should build the alias from one config object (dataset, model, n, splitter params, and soon the sparse model)



## NEXT STEPS

1. BM25 sparse retrieval plus Reciprocal Rank Fusion to merge it with my dense results. This is the "hybrid" in hybrid search and it's the first gap.
2. BGE cross-encoder reranker sitting after fusion, rescoring the merged candidates before they hit the LLM context window.
3. Citation formatting, so my FastAPI response returns structured JSON with source metadata attached to each answer chunk, not just the answer text.
4. The FastAPI layer itself with persisent checkpointer (which I already flagged) and per-user thread IDs.
5. Streaming response support on that endpoint.
6. The Ragas eval harness last, once I have a stable retrieval pipeline worth measuring.
7. The chunking pipeline already handles text. PDF and Markdown ingestion is a small extension (PyMuPDF + a Markdown loader) but it's listed in the spec.

