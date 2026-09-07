# Qdrant Collection Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an incomplete Qdrant index structurally impossible by having application code query an alias that is only ever pointed at a build whose upload completed.

**Architecture:** The name the app queries (`hf_docs__text_embedding_3_small__n50`) is a Qdrant alias, never a real collection. Each ingest writes to a uniquely timestamped build collection and swaps the alias onto it only after the upload is verified. A crashed ingest leaves an unaliased orphan that the next startup deletes before rebuilding. Sharing one `QdrantClient` forces abandoning `QdrantVectorStore`'s factory methods for its direct constructor, which is what makes explicit collection creation — and therefore aliasing — natural.

**Tech Stack:** Python 3.13, qdrant-client 1.19.0, langchain-qdrant 1.1.0, langchain-openai 1.6.0, LangGraph 1.2.11.

**Spec:** `docs/superpowers/specs/2026-09-06-qdrant-collection-integrity-design.md`

## Global Constraints

- Scope is `scripts/main.py` only. Do not touch the graph, REPL, tools, or `preprocess_dataset`.
- Run every command with `.venv/bin/python`. The system `python3` is 3.14 and has none of these packages installed.
- No new dependencies. A committed test suite is deferred by user decision and tracked in `NOTES.md` under Primary; each task below verifies with a throwaway script under `/tmp` that is deleted afterward.
- Never use `QdrantVectorStore.from_documents` or `QdrantVectorStore.from_existing_collection`. Neither accepts `client=`; `from_documents` forwards it into httpx and raises `TypeError: Client.__init__() got an unexpected keyword argument 'client'`. Use `QdrantVectorStore(client=..., collection_name=..., embedding=...)`.
- Alias format: `{base}__{model_slug}__n{number_of_docs}`, where `model_slug` is `embedding_model` lowercased with each run of non-alphanumeric characters replaced by a single underscore.
- Build collection format: `{alias}__build_{YYYYMMDDTHHMMSSZ}_{uuid4().hex[:8]}` in UTC. The random suffix is required: second-resolution timestamps collide when two ingests run in the same second, and `create_collection` raises `ValueError: Collection ... already exists`.
- Vector config is always `VectorParams(size=dim, distance=Distance.COSINE)`. `dim` comes from a single `embed_query` call, never hardcoded.
- Upload failures catch `BaseException`, not `Exception`, so `KeyboardInterrupt` is included.
- Progress output uses the existing `->`  prefix style.
- Verification scripts patch module attributes (`main.OpenAIEmbeddings`, `main.HuggingFaceDatasetLoader`) rather than adding injection parameters to production code.

---



### Task 1: Identity layer — client singleton and alias naming

**Files:**

- Modify: `scripts/main.py` (imports at lines 1-15, stubs `get_client` at line 36 and `alias_for` at line 40)

**Interfaces:**

- Consumes: module globals `qdrant_url`, `qdrant_key`, `embedding_model`, `number_of_docs`.
- Produces: `get_client() -> QdrantClient` (memoized singleton) and `alias_for(base: str) -> str`. Every later task calls both.

- [x] **Step 1: Commit the existing scaffold so later diffs are readable**

```bash
cd /Users/alexanderflores/agentic-rag
git add scripts/main.py
git commit -m "main: scaffold alias-swap ingest functions"
```

- [x] **Step 2: Add the required imports**

Add to the top of `scripts/main.py`, after `import os`:

```python
import re
import uuid
from datetime import datetime, timezone
```

Add after the existing `from qdrant_client import QdrantClient` line:

```python
from qdrant_client.http.models import (
    CreateAlias,
    CreateAliasOperation,
    Distance,
    VectorParams,
)
```

- [x] **Step 3: Implement** `get_client` **and** `alias_for`

Add a module-level `_client = None` immediately above `def get_client():`, then replace both stub bodies:

```python
_client = None

def get_client():
    '''returns a single module-level QdrantClient, created on first use and reused.'''
    global _client
    if _client is None:
        _client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    return _client

def alias_for(base):
    '''assembles the alias from base name, model slug, and document count.'''
    slug = re.sub(r"[^a-z0-9]+", "_", embedding_model.lower()).strip("_")
    return f"{base}__{slug}__n{number_of_docs}"
```

- [x] **Step 4: Verify**

Write `/tmp/verify_task1.py`:

```python
import sys
sys.path.insert(0, "scripts")
import main

assert main.alias_for("hf_docs") == "hf_docs__text_embedding_3_small__n50", main.alias_for("hf_docs")

main.embedding_model = "text-embedding-ada-002"
assert main.alias_for("hf_docs") == "hf_docs__text_embedding_ada_002__n50", main.alias_for("hf_docs")
main.embedding_model = "text-embedding-3-small"

main.number_of_docs = 500
assert main.alias_for("hf_docs") == "hf_docs__text_embedding_3_small__n500", main.alias_for("hf_docs")
main.number_of_docs = 50

assert main.get_client() is main.get_client(), "client is not memoized"
print("task 1 OK")
```

Run: `.venv/bin/python /tmp/verify_task1.py`
Expected: `task 1 OK`. A different model or doc count must change the alias; the client must be identical across calls.

- [x] **Step 5: Clean up and commit**

```bash
rm /tmp/verify_task1.py
git add scripts/main.py
git commit -m "main: add shared Qdrant client and model-aware alias naming"
```

---



### Task 2: Alias lifecycle — swap and orphan cleanup

**Files:**

- Modify: `scripts/main.py` (stubs `swap_alias` at line 48 and `cleanup_builds` at line 52)

**Interfaces:**

- Consumes: `alias_for` from Task 1; `CreateAlias`, `CreateAliasOperation` imports from Task 1.
- Produces: `swap_alias(client, alias, build) -> None` and `cleanup_builds(client, alias) -> None`. Task 4 calls both.

- [x] **Step 1: Implement** `swap_alias`

Replace the stub body. A bare `CreateAliasOperation` on an existing alias silently re-points it, so no delete is needed and the swap is one atomic call:

```python
def swap_alias(client, alias, build):
    '''points the alias at the completed build via one CreateAliasOperation.'''
    client.update_collection_aliases(
        change_aliases_operations=[
            CreateAliasOperation(
                create_alias=CreateAlias(collection_name=build, alias_name=alias)
            )
        ]
    )
```

- [x] **Step 2: Implement** `cleanup_builds`

Replace the stub body. Note the prefix is `__build_` with a single trailing underscore, matching Task 3's build names — the current stub docstring says `__build__`, so correct that too:

```python
def cleanup_builds(client, alias):
    '''deletes every {alias}__build_* collection that is not the alias's current target.'''
    prefix = f"{alias}__build_"
    current = None
    for entry in client.get_aliases().aliases:
        if entry.alias_name == alias:
            current = entry.collection_name
            break

    for collection in client.get_collections().collections:
        if collection.name.startswith(prefix) and collection.name != current:
            print(f"-> Removing stale build collection '{collection.name}'...")
            client.delete_collection(collection.name)
```

- [x] **Step 3: Verify**

Write `/tmp/verify_task2.py`:

```python
import sys
sys.path.insert(0, "scripts")
import main
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient(":memory:")
alias = main.alias_for("hf_docs")
names = [f"{alias}__build_A", f"{alias}__build_B", f"{alias}__build_C", "unrelated_collection"]
for name in names:
    client.create_collection(name, VectorParams(size=4, distance=Distance.COSINE))

# No alias yet: every build is an orphan, unrelated collections are untouched.
main.cleanup_builds(client, alias)
left = sorted(c.name for c in client.get_collections().collections)
assert left == ["unrelated_collection"], left

# With an alias, only the live target survives.
for name in (f"{alias}__build_A", f"{alias}__build_B"):
    client.create_collection(name, VectorParams(size=4, distance=Distance.COSINE))
main.swap_alias(client, alias, f"{alias}__build_B")
assert client.collection_exists(alias), "alias not resolvable"

main.cleanup_builds(client, alias)
left = sorted(c.name for c in client.get_collections().collections)
assert left == [f"{alias}__build_B", "unrelated_collection"], left

# Re-pointing an existing alias moves it in one operation.
client.create_collection(f"{alias}__build_D", VectorParams(size=4, distance=Distance.COSINE))
main.swap_alias(client, alias, f"{alias}__build_D")
target = [a.collection_name for a in client.get_aliases().aliases if a.alias_name == alias]
assert target == [f"{alias}__build_D"], target
print("task 2 OK")
```

Run: `.venv/bin/python /tmp/verify_task2.py`
Expected: `task 2 OK`. Orphans deleted, live target kept, unrelated collections never touched, alias re-point works.

- [x] **Step 4: Clean up and commit**

```bash
rm /tmp/verify_task2.py
git add scripts/main.py
git commit -m "main: add atomic alias swap and stale build cleanup"
```

---



### Task 3: Build collection creation and verified upload

**Files:**

- Modify: `scripts/main.py` (stub `ingest_collection` at line 44)

**Interfaces:**

- Consumes: `preprocess_dataset`, `HuggingFaceDatasetLoader`, `OpenAIEmbeddings`, `QdrantVectorStore`, `VectorParams`, `Distance`, `datetime`, `timezone`.
- Produces: `ingest_collection(client, alias, dataset) -> str` returning the build collection name. Task 4 passes that return value to `swap_alias`.

- [x] **Step 1: Implement** `ingest_collection`

Replace the stub body:

```python
def ingest_collection(client, alias, dataset):
    '''creates the build collection, uploads splits, returns the build name.'''
    print(f"-> Alias '{alias}' NOT found. Downloading and ingesting '{dataset}'...")
    loader = HuggingFaceDatasetLoader(dataset, "text")
    splits = preprocess_dataset(loader.load()[:number_of_docs])

    embeddings = OpenAIEmbeddings(model=embedding_model)
    dimension = len(embeddings.embed_query("dimension probe"))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    build = f"{alias}__build_{stamp}_{uuid.uuid4().hex[:8]}"
    client.create_collection(
        build, VectorParams(size=dimension, distance=Distance.COSINE)
    )

    try:
        store = QdrantVectorStore(
            client=client, collection_name=build, embedding=embeddings
        )
        store.add_documents(splits)
        uploaded = client.get_collection(build).points_count
        if uploaded != len(splits):
            raise RuntimeError(
                f"expected {len(splits)} points in '{build}', found {uploaded}"
            )
    except BaseException:
        print(f"-> Ingest failed. Discarding incomplete build '{build}'...")
        client.delete_collection(build)
        raise

    print(f"-> Uploaded {len(splits)} chunks to '{build}'.")
    return build
```

- [ ] **Step 2: Verify the success path and the crash path**

Write `/tmp/verify_task3.py`:

```python
import sys
sys.path.insert(0, "scripts")
import main
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding


class FakeLoader:
    def __init__(self, path, page_content_column):
        self.path = path

    def load(self):
        return [Document(page_content=f"transformers pipeline note {i}") for i in range(8)]


main.HuggingFaceDatasetLoader = FakeLoader
main.OpenAIEmbeddings = lambda model=None: DeterministicFakeEmbedding(size=8)

client = QdrantClient(":memory:")
alias = main.alias_for("hf_docs")

build = main.ingest_collection(client, alias, "fake/dataset")
assert build.startswith(f"{alias}__build_"), build
assert client.get_collection(build).points_count == 8, client.get_collection(build).points_count
assert not client.collection_exists(alias), "ingest must not create the alias itself"

# Crash path: a failing upload must leave no collection behind.
real_store = main.QdrantVectorStore

class ExplodingStore:
    def __init__(self, **kwargs):
        pass

    def add_documents(self, docs):
        raise KeyboardInterrupt("simulated Ctrl-C mid-upload")

main.QdrantVectorStore = ExplodingStore
before = {c.name for c in client.get_collections().collections}
try:
    main.ingest_collection(client, alias, "fake/dataset")
except KeyboardInterrupt:
    pass
else:
    raise AssertionError("expected the interrupt to propagate")
after = {c.name for c in client.get_collections().collections}
assert before == after, f"partial build left behind: {after - before}"
main.QdrantVectorStore = real_store
print("task 3 OK")
```

Run: `.venv/bin/python /tmp/verify_task3.py`
Expected: `task 3 OK`. The success path uploads 8 points and does not create the alias; the interrupted path propagates `KeyboardInterrupt` and leaves zero new collections.

- [ ] **Step 3: Clean up and commit**

```bash
rm /tmp/verify_task3.py
git add scripts/main.py
git commit -m "main: build into timestamped collection with verified upload"
```

---



### Task 4: Orchestration and call-site migration

**Files:**

- Modify: `scripts/main.py` (stub `get_or_create_retriever` at line 56, commented-out block at lines 61-93, `ingest` at line 95)

**Interfaces:**

- Consumes: `alias_for`, `cleanup_builds`, `ingest_collection`, `swap_alias`, `get_client`.
- Produces: `get_or_create_retriever(client, base, dataset) -> VectorStoreRetriever`. `ingest()` calls it twice and passes the retrievers to `create_retriever_tool` unchanged.

- [ ] **Step 1: Implement** `get_or_create_retriever`

Replace the stub body. Cleanup runs first and unconditionally, so a stale build is cleared whether or not a working alias exists:

```python
def get_or_create_retriever(client, base, dataset):
    '''orchestrates cleanup, then load or build.'''
    alias = alias_for(base)
    cleanup_builds(client, alias)

    if client.collection_exists(alias):
        print(f"-> Alias '{alias}' found in Qdrant. Loading existing index...")
    else:
        build = ingest_collection(client, alias, dataset)
        swap_alias(client, alias, build)
        cleanup_builds(client, alias)
        print(f"-> Alias '{alias}' now points at '{build}'.")

    store = QdrantVectorStore(
        client=client,
        collection_name=alias,
        embedding=OpenAIEmbeddings(model=embedding_model),
    )
    return store.as_retriever()
```

- [ ] **Step 2: Delete the commented-out legacy functions**

Remove the entire commented block spanning `# def create_retriever(collection_name, doc_splits):` through `#         return create_retriever(collection_name, splits)` (lines 61-93). It is superseded and the spec forbids both factory methods it uses.

- [ ] **Step 3: Update** `ingest` **to pass the shared client**

Replace the first two lines of `ingest`:

```python
def ingest():
    client = get_client()
    hf_retriever = get_or_create_retriever(client, "hf_docs", "m-ric/huggingface_doc")
    transformer_retriever = get_or_create_retriever(
        client, "transformer_docs", "m-ric/transformers_documentation_en"
    )
```

Leave the two `create_retriever_tool` calls and the return statement exactly as they are.

- [ ] **Step 4: Verify the full lifecycle in memory**

Write `/tmp/verify_task4.py`:

```python
import sys
sys.path.insert(0, "scripts")
import main
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding


class FakeLoader:
    def __init__(self, path, page_content_column):
        self.path = path

    def load(self):
        return [Document(page_content=f"transformers pipeline note {i}") for i in range(8)]


main.HuggingFaceDatasetLoader = FakeLoader
main.OpenAIEmbeddings = lambda model=None: DeterministicFakeEmbedding(size=8)

client = QdrantClient(":memory:")
alias = main.alias_for("hf_docs")

# Cold start: builds, aliases, and returns a working retriever.
retriever = main.get_or_create_retriever(client, "hf_docs", "fake/dataset")
assert client.collection_exists(alias), "alias missing after cold start"
first = [a.collection_name for a in client.get_aliases().aliases if a.alias_name == alias][0]
assert retriever.invoke("pipeline"), "retriever returned nothing"

# Warm start: reuses the same build, no new collection.
main.get_or_create_retriever(client, "hf_docs", "fake/dataset")
second = [a.collection_name for a in client.get_aliases().aliases if a.alias_name == alias][0]
assert first == second, f"rebuilt unnecessarily: {first} -> {second}"
builds = [c.name for c in client.get_collections().collections if c.name.startswith(f"{alias}__build_")]
assert builds == [first], builds

# Crashed ingest: an unaliased orphan is cleaned up on the next startup.
orphan = f"{alias}__build_19990101T000000Z"
client.create_collection(orphan, VectorParams(size=8, distance=Distance.COSINE))
main.get_or_create_retriever(client, "hf_docs", "fake/dataset")
assert not client.collection_exists(orphan), "orphan build survived cleanup"
assert client.collection_exists(alias), "cleanup destroyed the live alias"

# Changing the model produces a separate alias and leaves the old one intact.
main.embedding_model = "text-embedding-ada-002"
other = main.alias_for("hf_docs")
assert other != alias
main.get_or_create_retriever(client, "hf_docs", "fake/dataset")
assert client.collection_exists(other) and client.collection_exists(alias)
main.embedding_model = "text-embedding-3-small"
print("task 4 OK")
```

Run: `.venv/bin/python /tmp/verify_task4.py`
Expected: `task 4 OK`. This covers four of the five deferred test scenarios from the spec.

- [ ] **Step 5: Confirm the module still imports cleanly**

Run: `.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); import main; print('import OK')"`
Expected: `import OK` with no `NameError` from the deleted legacy block.

- [ ] **Step 6: Clean up and commit**

```bash
rm /tmp/verify_task4.py
git add scripts/main.py
git commit -m "main: orchestrate alias-backed retrieval and drop legacy ingest path"
```

---



### Task 5: Live acceptance run against Qdrant Cloud

**Files:**

- Modify: none. This task only runs the program.

**Interfaces:**

- Consumes: everything from Tasks 1-4.
- Produces: a populated Qdrant instance and confirmation that the warm path skips ingestion.

This is the first task that spends money — two datasets of 50 documents through `text-embedding-3-small`, on the order of cents. The Qdrant instance currently holds zero collections, so this is a genuine cold start.

- [ ] **Step 1: Cold run**

Run: `.venv/bin/python scripts/main.py`
Expected: two `-> Alias '...' NOT found. Downloading and ingesting ...` lines, each followed by `-> Uploaded N chunks to '...__build_<timestamp>'.` and `-> Alias '...' now points at '...'`. Then the `Ready.` prompt.

- [ ] **Step 2: Confirm retrieval works, then exit**

At the `User:` prompt, ask: `What does the transformers pipeline function do?`
Expected: a grounded answer. Then type `quit`.

- [ ] **Step 3: Warm run**

Run: `.venv/bin/python scripts/main.py`
Expected: two `-> Alias '...' found in Qdrant. Loading existing index...` lines, no download, no upload, and a noticeably faster start.

- [ ] **Step 4: Inspect the resulting server state**

Write `/tmp/verify_task5.py`:

```python
import sys
sys.path.insert(0, "scripts")
import main

client = main.get_client()
aliases = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
print("aliases:", aliases)

expected = {}
for base in ("hf_docs", "transformer_docs"):
    alias = main.alias_for(base)
    assert alias in aliases, f"missing alias {alias}"
    target = aliases[alias]
    assert target.startswith(f"{alias}__build_"), target
    points = client.get_collection(target).points_count
    assert points > 0, f"{target} is empty"
    expected[alias] = target
    print(f"  {alias} -> {target}, points={points}")

# Only builds belonging to our two aliases should exist, one apiece.
ours = [
    c.name
    for c in client.get_collections().collections
    if any(c.name.startswith(f"{a}__build_") for a in expected)
]
assert sorted(ours) == sorted(expected.values()), f"stale builds present: {set(ours) - set(expected.values())}"
print("task 5 OK")
```

Run: `.venv/bin/python /tmp/verify_task5.py`
Expected: `task 5 OK`, with exactly two build collections, each carrying a non-zero point count and each the target of its alias. No leftovers.

- [ ] **Step 5: Clean up**

```bash
rm /tmp/verify_task5.py
```

- [ ] **Step 6: Mark the URGENT items resolved in** `NOTES.md`

Delete the three entries under `## URGENT` (`**Collection.**`, `**Collection Embedding Model.**`, `**Client on Every Call.**`) and the now-empty heading, since all three are closed by this plan. Leave the Primary, Secondary, and NEXT STEPS sections untouched.

```bash
git add NOTES.md
git commit -m "personal: clear resolved URGENT items"
```

---



## Known follow-ups

Deliberately out of scope, already tracked in `NOTES.md`:

- The committed test suite. The verification scripts above prove the behaviour but are deleted as they go. When the suite is written, note that `ingest_collection` constructs its own loader and embeddings, so tests will patch `main.HuggingFaceDatasetLoader` and `main.OpenAIEmbeddings` exactly as the scripts here do.
- `.load()[:number_of_docs]` still materializes the full dataset before slicing. This plan preserves that behaviour rather than fixing it, so the change stays reviewable.
- Environment variable validation. `get_client()` is the natural home for it when that item comes up.
- Concurrent ingest from separate processes. The uuid suffix stops two simultaneous builds from colliding on a name, but two workers booting at once would each build a full index, both swap the alias, and each `cleanup_builds` could delete the other's in-flight build. Harmless for a single-process CLI; revisit when the FastAPI layer runs multiple workers.

