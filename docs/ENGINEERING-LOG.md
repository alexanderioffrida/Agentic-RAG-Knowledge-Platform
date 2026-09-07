# Engineering Log

Running record of what this project is, what works today, the design decisions behind it, and
what is deliberately still open. Written so someone can pick the work up cold.

Last updated: 2026-09-07

## The project

A hybrid-search RAG service that routes queries across retrieval, a web fallback, and a reranker,
returning grounded answers with citations, backed by an offline eval harness that scores
faithfulness.

Dense vector search (Qdrant) merges with BM25 sparse retrieval via Reciprocal Rank Fusion. A
LangGraph agent decides whether to answer from the index, fall back to live web search, or
escalate to a human. A BGE cross-encoder reranks candidate passages before they reach the LLM
context window. FastAPI serves structured JSON with inline citations and confidence metadata, with
token streaming. Ragas scores faithfulness, context precision/recall, and answer relevancy offline
against a golden QA set.

**Stack:** LangGraph · Qdrant · FastAPI · BGE reranker · Ragas · sentence-transformers · OpenRouter

## What works today

A single-file CLI agent at `scripts/main.py` (231 lines), running against a live Qdrant Cloud
instance.

- **Ingestion.** Two HuggingFace documentation datasets are chunked at 700 tokens with 50 tokens
  of overlap, embedded with `text-embedding-3-small`, and uploaded to Qdrant. Currently indexed:
  281 chunks from `m-ric/huggingface_doc`, 386 from `m-ric/transformers_documentation_en`.
- **Idempotent, crash-safe reindexing.** Startup reuses an existing index with no download and no
  embedding spend. A crashed ingest cannot corrupt or silently degrade the live index. See
  "Index integrity" below — this is the most substantive engineering in the project so far.
- **Agent.** A LangGraph `StateGraph` with an agent node and a tool node, looping until the model
  stops requesting tools. Three tools: two retrievers (one per dataset) and Brave web search.
- **Multi-turn memory.** `MemorySaver` checkpointer with a `thread_id`, so follow-up questions
  resolve against conversation history.
- **REPL.** Streams agent events, announcing each tool call as it fires and printing only
  non-empty assistant content.

Run it with `.venv/bin/python scripts/main.py` from the repo root. Python 3.13 in `.venv`; the
system `python3` is 3.14 and has none of the dependencies.

## Index integrity: the alias-swap design

The problem: `QdrantVectorStore.from_documents` creates a collection *before* it finishes
uploading points. An ingest interrupted by Ctrl-C, a rate limit, or a dropped connection leaves a
collection that exists but holds a fraction of the chunks. Since the startup check was
`collection_exists(name)`, every later run would take the "found" branch and silently serve an
incomplete index, with nothing in the output to indicate it.

The fix inverts the question. Instead of *detecting* incompleteness, make it unrepresentable.

The name application code queries — `hf_docs__text_embedding_3_small__n50` — is a Qdrant **alias**,
never a real collection. Each ingest writes to a uniquely named build collection
(`{alias}__build_{timestamp}_{uuid8}`) and points the alias at it only after the upload completes
and its point count is verified. A crashed ingest leaves an orphan that no alias references, so
`collection_exists(alias)` is `False` and the next startup rebuilds. Orphans are swept on every
startup before the existence check.

**The invariant: the alias only ever points at a build whose upload succeeded.** Completeness is
structural, not measured — there is no chunk count to store, no manifest to keep honest, and no
second source of truth that can drift.

Design spec: `docs/superpowers/specs/2026-09-06-qdrant-collection-integrity-design.md`
Implementation plan: `docs/superpowers/plans/2026-09-06-qdrant-collection-integrity.md`

## Design decisions

**Alias indirection over a completeness check.** Three alternatives were prototyped and rejected.
A sentinel marker point inside the data collection was killed by evidence: a probe showed the
sentinel returned as a ranked similarity-search hit, so it would pollute retrieval unless every
query carried a filter. A separate manifest collection recording expected chunk counts worked, but
kept two sources of truth that can disagree and required `wait=True` upsert semantics to avoid
reading stale counts. A local JSON manifest was simplest but is per-machine, lies if the cluster
is rebuilt elsewhere, and becomes actively wrong once FastAPI runs multiple workers. Aliasing was
chosen because it is the only option where an incomplete index cannot be represented at all, and
because it doubles as the zero-downtime reindex mechanism needed when BM25 sparse vectors are
added.

**Index identity is encoded in the name.** The alias carries both the embedding model and the
document count. Without the model, changing embeddings would query old vectors with new
embeddings — and because `text-embedding-3-small` and `ada-002` are both 1536-dimensional, that
fails *silently* rather than raising a dimension error. Without the document count, raising
`number_of_docs` from 50 to 500 would silently reuse the smaller index. Both are now impossible:
either change produces a new alias and a fresh build, leaving the old index intact.

**Vector dimension is derived, never hardcoded.** A single `embed_query` call at build time
determines the dimension. Hardcoding 1536 would have reintroduced exactly the model coupling the
naming scheme exists to remove.

**`except BaseException`, not `except Exception`.** `KeyboardInterrupt` derives from
`BaseException`, and Ctrl-C mid-upload is the single most likely interruption. Catching only
`Exception` would have left the exact orphan the design exists to prevent.

**Cleanup runs before the existence check, unconditionally.** A stale build is swept whether or
not a working alias is present, so a failed run never leaves cost behind.

**Failure is auto-repairing rather than fatal.** An incomplete ingest simply has no alias, so the
next run rebuilds through the normal path. There is no special-case recovery branch.

## Unexpected findings

Every item here was verified empirically against the installed versions rather than assumed.
Several changed the design.

**LangChain's Qdrant factory methods cannot share a client.** Neither `from_documents` nor
`from_existing_collection` accepts `client=`; the former forwards it into httpx and raises
`TypeError: Client.__init__() got an unexpected keyword argument 'client'`. Only the direct
constructor `QdrantVectorStore(client=..., ...)` takes one. This turned "reuse one connection"
from a one-line change into a restructure — and that restructure, which forces explicit collection
creation, is what made alias swapping nearly free. The two problems solved each other.

**Qdrant aliases resolve transparently through the entire stack.** `collection_exists`,
`get_collection`, `query_points`, `QdrantVectorStore`, and the retriever built from it all accept
an alias wherever a collection name is expected. No adapter layer was needed.

**Re-pointing an existing alias is atomic and silent.** A bare `CreateAliasOperation` on a live
alias moves it without error, so the swap is one operation with no delete-then-create window.

**Deleting a collection automatically drops its alias.** Cleanup therefore cannot strand a
dangling pointer, which removed an entire class of failure from the design.

**Second-resolution timestamps collide.** The build name originally used
`strftime("%Y%m%dT%H%M%SZ")`. Two ingests inside the same wall-clock second produced byte-identical
names and the second `create_collection` raised `ValueError: Collection ... already exists`. The
plan's own verification script caught this before it reached production. Fixed with a `uuid4`
suffix, chosen over microsecond precision because clock-based names give no protection once
multiple FastAPI workers can ingest concurrently.

**The dimension probe doubles as a preflight check.** When the OpenAI account hit
`insufficient_quota` mid-run, the failure landed on `embed_query` — *before* `create_collection`.
No collection was created, no alias touched, nothing to clean up by hand. The crash path was
validated for real by an unplanned outage. The probe currently runs after the dataset download,
so moving it earlier would fail in a second instead of after a 22MB pull.

**The `langchain` package no longer exists in this stack.** LangChain 1.x split into
`langchain-core`, `langchain-openai`, `langchain-classic`, and friends; the umbrella package isn't
installed at all. Every `from langchain.x import y` in the original draft was a hard failure.

**`@tool` has strict requirements.** It raises `ValueError` outright if the decorated function has
no docstring, and an unannotated argument yields a JSON schema property with no `type`, which
strict function-calling rejects.

## Verification approach

There is no committed test suite yet. During implementation, each task was verified with a
throwaway script using `QdrantClient(":memory:")` — a real local Qdrant — plus
`DeterministicFakeEmbedding` in place of OpenAI, so the full lifecycle could be exercised with no
cloud calls and no embedding spend. Scenarios covered: cold build and alias creation, warm reuse
without re-embedding, simulated `KeyboardInterrupt` mid-upload leaving zero residue, orphan
cleanup, and an embedding-model change producing a separate alias.

Those scripts were deleted as each task closed. **Porting them into a committed suite is the
largest outstanding quality gap**, and it matters more now that the retrieval pipeline is about to
get significantly more complex.

## Open items

### The structural decision to make first

The current graph is an agent-with-three-tools loop: the LLM freely picks a retriever or Brave,
reads the raw result, and answers. The target architecture is a pipeline — hybrid retrieval and
RRF produce a candidate pool, a confidence check decides whether to pull in web results, and
everything flows through the BGE reranker before reaching the context window.

Two consequences follow, and both should be settled before the reranker is built:

1. **Web results must join the candidate pool**, not bypass it. The brief places the web fallback
   *before* reranking, so web passages should be rescored alongside Qdrant hits. Today web search
   is a tool whose output the model reads directly, skipping reranking entirely.
2. **"Low confidence" needs a real signal.** Gating the fallback on RRF or rerank scores against a
   threshold is a different mechanism from letting the LLM decide, which is what happens now.

### Remaining deliverables

1. BM25 sparse retrieval merged with dense results via Reciprocal Rank Fusion. Requires rebuilding
   collections with named sparse vectors — the alias swap exists precisely to make this safe.
   `fastembed` is not yet installed.
2. BGE cross-encoder reranker after fusion. `sentence-transformers` 6.0.1 and `torch` 2.14.0 are
   already installed.
3. Citation formatting. `create_retriever_tool(..., response_format="content_and_artifact")`
   preserves source metadata instead of flattening it to a string.
4. FastAPI endpoint returning structured JSON with citations **and confidence metadata**.
5. Token streaming on that endpoint.
6. Ragas offline eval harness scoring faithfulness, context precision, context recall, and answer
   relevancy against a golden QA set. Last, once retrieval is stable enough to be worth measuring.
7. PDF and Markdown ingestion (PyMuPDF plus a Markdown loader) alongside the existing text path.
8. **Human escalation** as a third routing branch. Named in the brief, not yet tracked anywhere
   else.

### Carried over from NOTES.md

- **FastAPI persistence.** `MemorySaver` is in-process only; history dies with the REPL. The
  service layer needs a durable checkpointer and a `thread_id` per user rather than a constant.
- **Unbounded history.** One hardcoded `thread_id` accumulates every retrieved chunk, so long
  sessions drift toward the context limit and rising per-turn cost.
- No system prompt directing the agent between the two retrievers and web search.
- `.load()[:50]` materializes the entire dataset before slicing.
- `route` is a hand-rolled reimplementation of `langgraph.prebuilt.tools_condition`.
- No environment variable validation; `get_client()` is the natural home.
- No type hints on `preprocess_dataset` and friends.
- No `requirements.txt` or `pyproject.toml`, so the environment isn't reproducible. Note that
  `dotenv` is installed alongside `python-dotenv`; only the latter belongs in a dependency list,
  as `dotenv` is a deprecated redirect stub.

### Known limitations accepted for now

- **Concurrent ingest across processes.** The uuid suffix prevents name collisions, but two
  workers booting simultaneously would each build a full index, both swap the alias, and each
  cleanup pass could delete the other's in-flight build. Harmless for a single-process CLI;
  revisit with the FastAPI layer.
- **`langchain-community` is sunsetting.** The dataset loader and Brave tool both come from it and
  emit deprecation warnings. Replacements are `datasets` directly and a plain HTTP call to Brave.
