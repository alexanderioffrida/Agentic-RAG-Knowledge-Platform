# Qdrant Collection Integrity via Alias Swap

Date: 2026-09-06
Status: Approved, pending implementation
Scope: `scripts/main.py` ingestion path only

## Problem

`get_or_create_retriever` decides whether to ingest by calling `client.collection_exists(name)`.
That signal is too coarse. `QdrantVectorStore.from_documents` creates the collection before it
finishes uploading points, so an ingest interrupted by Ctrl-C, an OpenAI rate limit, or a dropped
connection leaves a collection that exists but holds a fraction of the chunks. Every subsequent
run then takes the "found" branch and silently serves an incomplete index, with nothing in the
output to indicate it.

Two related defects share the same function:

- The collection name does not encode the embedding model. Changing `embedding_model` would query
  old vectors with new embeddings. Because `text-embedding-3-small` and `ada-002` are both
  1536-dimensional, this fails silently rather than raising a dimension error.
- `get_or_create_retriever` constructs a `QdrantClient` on every call and never closes it, while
  `from_existing_collection` constructs another internally.

These correspond to the three URGENT items in `NOTES.md`.

## Verified constraints

Established empirically against the installed versions (`qdrant-client` 1.19.0,
`langchain-qdrant` 1.1.0) using `QdrantClient(":memory:")`:

1. Neither `QdrantVectorStore.from_documents` nor `QdrantVectorStore.from_existing_collection`
   accepts a `client=` argument. `from_existing_collection` has no such parameter, and
   `from_documents` forwards it into httpx, raising
   `TypeError: Client.__init__() got an unexpected keyword argument 'client'`.
   The direct constructor `QdrantVectorStore(client=..., collection_name=..., embedding=...)`
   does accept one and retains it (`vs.client is c`).
2. Aliases resolve transparently throughout the stack: `collection_exists`, `get_collection`,
   `query_points`, `QdrantVectorStore`, and the retriever built from it all accept an alias
   where a collection name is expected.
3. Re-pointing an existing alias with a single `CreateAliasOperation` silently moves it. No
   delete-then-create sequence is required, so the swap is one atomic operation.
4. Deleting an alias that does not exist does not raise.
5. Deleting a collection automatically removes any alias pointing at it, after which
   `collection_exists(alias)` returns `False`. Cleanup cannot strand a dangling alias.
6. A sentinel marker point stored inside a data collection is returned as a ranked similarity
   search hit. This rules out the in-collection sentinel variant unless every query is filtered.

## Approach

The name application code queries is an **alias**, never a real collection. Ingest writes to a
uniquely named **build collection** and points the alias at it only after the upload returns
successfully.

The resulting invariant: *the alias only ever points at a build whose upload completed.*
Index completeness is structurally guaranteed rather than measured, so there is no chunk count
to compare, no manifest to maintain, and no second source of truth that can drift.

This also supports the planned migration to hybrid retrieval (`NOTES.md` NEXT STEPS #1), which
requires rebuilding collections with named sparse vectors. A new index can be built alongside the
live one and swapped in atomically.

## Naming

Alias, which encodes everything determining index identity:

```
{base}__{model_slug}__n{number_of_docs}
hf_docs__text_embedding_3_small__n50
```

Build collection, unique per ingest run:

```
{alias}__build_{utc_timestamp}
hf_docs__text_embedding_3_small__n50__build_20260906T213000Z
```

`model_slug` lowercases `embedding_model` and replaces each run of non-alphanumeric characters
with a single underscore, so `text-embedding-3-small` becomes `text_embedding_3_small`.

`number_of_docs` is included so that raising it from 50 to 500 produces a new alias and a fresh
build, rather than silently reusing the smaller index. This is the same class of defect as the
partial-ingest bug and is closed the same way.

## Components

Replacing `create_retriever`, `get_retriever`, and `get_or_create_retriever`:

| Function | Responsibility |
| --- | --- |
| `get_client()` | Returns a single module-level `QdrantClient`, created on first use and reused. |
| `alias_for(base)` | Assembles the alias from base name, model slug, and document count. |
| `ingest_collection(client, alias, dataset)` | Creates the build collection, uploads splits, returns the build name. |
| `swap_alias(client, alias, build)` | Points the alias at the completed build via one `CreateAliasOperation`. |
| `cleanup_builds(client, alias)` | Deletes every `{alias}__build_*` collection that is not the alias's current target. |
| `get_or_create_retriever(client, base, dataset)` | Orchestrates cleanup, then load or build. |

`ingest_collection` creates the collection explicitly with
`VectorParams(size=dim, distance=Distance.COSINE)`. Cosine matches the default that
`langchain-qdrant` applies in both `from_documents` and `from_existing_collection`, so retrieval
behaviour is unchanged from the current code. The dimension is derived at build time from a
single `embed_query` call rather than hardcoded to 1536; hardcoding would reintroduce the model
coupling this design removes.

Before swapping, `ingest_collection` asserts
`client.get_collection(build).points_count == len(splits)`. This is a one-off check against a
count already in hand, not the recurring startup comparison rejected during design, and it
guarantees the upload is durably visible before the alias makes it live.

`cleanup_builds` resolves the alias's current target by listing aliases and matching on
`alias_name`, then deletes every collection whose name starts with `{alias}__build_` except that
target. When no alias exists, every matching build is an orphan and all are deleted.

`preprocess_dataset` and the two `create_retriever_tool` calls in `ingest()` are unchanged.

## Control flow

For each `(base, dataset)` pair at startup:

1. Call `cleanup_builds(client, alias)` unconditionally, clearing orphans from any previous crash.
2. If `collection_exists(alias)`, construct `QdrantVectorStore(client=..., collection_name=alias,
   embedding=...)` and return its retriever. No download, no embedding spend.
3. Otherwise load and split the dataset, create the build collection, upload, `swap_alias`, then
   `cleanup_builds` again to drop the superseded build.

Cleanup runs before the existence check so that a stale build is removed whether or not a working
alias is present.

## Error handling

The create-and-upload sequence is wrapped in `try` / `except BaseException`, which deletes the
partial build collection and re-raises. `BaseException` rather than `Exception` because
`KeyboardInterrupt` is precisely the interruption of concern.

A hard kill that bypasses Python leaves an orphan build collection, which the next startup's
`cleanup_builds` removes before rebuilding.

In every failure path the alias is left untouched, so a failed re-index can never degrade an
index that already works.

The auto-rebuild behaviour chosen during design is implicit rather than a special case: an
incomplete ingest has no alias, so the next run naturally takes the build branch. Both branches
print progress in the existing `->` log style.

## Testing

Deferred by decision during design and tracked in `NOTES.md` under Primary.

When implemented, `QdrantClient(":memory:")` provides a real local Qdrant and
`DeterministicFakeEmbedding` substitutes for OpenAI, so the full lifecycle is testable with no
cloud calls and no embedding cost. Five scenarios:

1. Cold start builds the collection and creates the alias.
2. Warm start reuses the existing alias without re-embedding.
3. An unaliased build collection, simulating a crashed ingest, is cleaned up and rebuilt.
4. Changing the embedding model yields a new alias and leaves the previous one intact.
5. Multiple stale builds are all collected except the one the alias points at.

## Out of scope

Deliberately excluded to keep this change reviewable:

- Environment variable validation.
- The routing system prompt.
- Replacing `route` with `tools_condition`.
- `.load()[:n]` materializing the full dataset before slicing.
- Hybrid retrieval, the BGE reranker, and citations.

## Migration

None required. The Qdrant instance currently holds zero collections, so there is no existing data
to rename or move.
