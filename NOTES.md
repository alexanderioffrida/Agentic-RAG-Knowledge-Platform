# NOTES

Personal notes on urgent items to be fixed / implemented.

## URGENT
**Collection.** `collection_exists` is too coarse a signal, as `from_documents` creates the collection before it finishes uploading points, so if the first ingest dies partway i'm left with a collection that *exists* but holds a *fraction* of the chunks. every subsequent run then takes the "found" branch and silently serves an incomplete index, and nothing in the output would tell me. checking `get_collection(name).points_count` against the chunk count i'd expect, or writing a sentinel point after a successful upload, would close this gap.
**Collection Embedding Model.** the collection name doesn't encode the embedding model, so if `embedding_model` ever changes, i'll load vectors from the old model against queries from the new one –– and since `text-embedding-3-small` and `ada-002` are both 1536-dimensional, that fails silently rather than loudly. i can fold the model into the collection name to avoid this.
**Client on Every Call.** `get_or_create_retriever` builds a `QdrantClient` on every call and never closes it, while `from_existing_collection` builds another one internally. passing `client=` through would avoid the duplicate connections.

## Primary
**Ingest Tests.** deferred from the alias-swap design, so the whole ingest lifecycle currently ships untested. `QdrantClient(":memory:")` gives a real local Qdrant and `DeterministicFakeEmbedding` stands in for OpenAI, so this costs nothing to run: cold start builds and aliases, warm start reuses without re-embedding, an unaliased build gets cleaned and rebuilt, a model change yields a new alias, stale builds get collected. spec is at `docs/superpowers/specs/2026-09-06-qdrant-collection-integrity-design.md`.
**REPL #1.** `run_agent` only prints on `agent` events, which means there's zero feedback while a retrieval or Brave search is in flight. a one-line notice on tool events would fix this without reintroducing JSON noise.
**FUTURE: FastAPI Layer.** `MemorySaver` is in-process only, so history dies with the REPL. when i get to the FastAPI layer i'll want a persistent checkpointer and a `thread_id` per user rather than a constant.

## Secondary
**REPL #2.** with a single hardcoded `thread_id`, history grows without bound, including every 700-token retrieved chunk, so a long session will creep toward the context limit and rising per-turn cost.

- No system prompt to drive routing between the two retrievers and web search.
- `.load()[:50]` still materializes the whole dataset before slicing.
- `route` is still a manual reimplementation of `tools_condition`.
- No env-var validation.
- No type hints on `preprocess_dataset`/`create_retriever`.
- Hybrid retrieval, the BGE ranker, and citations are all still absent.
## NEXT STEPS
1. BM25 sparse retrieval plus Reciprocal Rank Fusion to merge it with my dense results. This is the "hybrid" in hybrid search and it's the first gap.
2. BGE cross-encoder reranker sitting after fusion, rescoring the merged candidates before they hit the LLM context window.
3. Citation formatting, so my FastAPI response returns structured JSON with source metadata attached to each answer chunk, not just the answer text.
4. The FastAPI layer itself with persisent checkpointer (which I already flagged) and per-user thread IDs.
5. Streaming response support on that endpoint.
6. The Ragas eval harness last, once I have a stable retrieval pipeline worth measuring.
7. The chunking pipeline already handles text. PDF and Markdown ingestion is a small extension (PyMuPDF + a Markdown loader) but it's listed in the spec.