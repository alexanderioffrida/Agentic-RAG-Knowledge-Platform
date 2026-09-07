import pytest
from qdrant_client.http.models import Distance, VectorParams
from scripts import main


def test_alias_naming():
    '''alias_for correctly encodes base name, normalized model slug, and doc count.'''
    assert main.alias_for("hf_docs") == "hf_docs__text_embedding_3_small__n50"

    saved_model = main.embedding_model
    try:
        main.embedding_model = "text-embedding-ada-002"
        assert main.alias_for("hf_docs") == "hf_docs__text_embedding_ada_002__n50"

        main.embedding_model = "BAAI/bge-m3"
        assert main.alias_for("hf_docs") == "hf_docs__baai_bge_m3__n50"
    finally:
        main.embedding_model = saved_model

    saved_docs = main.number_of_docs
    try:
        main.number_of_docs = 500
        assert main.alias_for("hf_docs") == "hf_docs__text_embedding_3_small__n500"
    finally:
        main.number_of_docs = saved_docs


def test_client_singleton():
    '''get_client returns the same memoized QdrantClient instance.'''
    client1 = main.get_client()
    client2 = main.get_client()
    assert client1 is client2


def test_alias_swap_and_resolution(memory_client):
    '''swap_alias creates the alias and resolves transparently through collection_exists.'''
    client = memory_client
    alias = main.alias_for("hf_docs")
    build_a = f"{alias}__build_A"

    client.create_collection(build_a, VectorParams(size=8, distance=Distance.COSINE))
    assert not client.collection_exists(alias)

    main.swap_alias(client, alias, build_a)
    assert client.collection_exists(alias)

    aliases = [a.collection_name for a in client.get_aliases().aliases if a.alias_name == alias]
    assert aliases == [build_a]


def test_cleanup_builds_removes_orphans_and_preserves_target(memory_client):
    '''cleanup_builds removes orphaned builds when unaliased, and sweeps old builds when aliased.'''
    client = memory_client
    alias = main.alias_for("hf_docs")

    build_a = f"{alias}__build_A"
    build_b = f"{alias}__build_B"
    build_c = f"{alias}__build_C"
    unrelated = "unrelated_collection"

    for name in (build_a, build_b, build_c, unrelated):
        client.create_collection(name, VectorParams(size=8, distance=Distance.COSINE))

    # without an alias, every matching build is an orphan and swept; unrelated is untouched
    main.cleanup_builds(client, alias)
    collections = sorted(c.name for c in client.get_collections().collections)
    assert collections == [unrelated]

    # create two more builds and point the alias at build_e
    build_d = f"{alias}__build_D"
    build_e = f"{alias}__build_E"
    client.create_collection(build_d, VectorParams(size=8, distance=Distance.COSINE))
    client.create_collection(build_e, VectorParams(size=8, distance=Distance.COSINE))
    main.swap_alias(client, alias, build_e)

    # cleanup should keep only the alias target (build_e) and unrelated
    main.cleanup_builds(client, alias)
    collections = sorted(c.name for c in client.get_collections().collections)
    assert collections == [build_e, unrelated]


def test_ingest_collection_success(memory_client, patch_main_dependencies):
    '''ingest_collection creates timestamped build, verifies points count, and returns build name.'''
    client = memory_client
    alias = main.alias_for("hf_docs")

    build = main.ingest_collection(client, alias, "fake/dataset")
    assert build.startswith(f"{alias}__build_")

    assert client.count(build, exact=True).count == 8
    assert not client.collection_exists(alias)


def test_ingest_collection_crash_leaves_zero_residue(memory_client, patch_main_dependencies, monkeypatch):
    '''simulated interruption during upload cleans up the incomplete build and re-raises.'''
    client = memory_client
    alias = main.alias_for("hf_docs")

    class ExplodingVectorStore:
        def __init__(self, **kwargs):
            pass

        def add_documents(self, docs):
            raise KeyboardInterrupt("simulated Ctrl-C mid-upload")

    monkeypatch.setattr(main, "QdrantVectorStore", ExplodingVectorStore)

    before_collections = {c.name for c in client.get_collections().collections}
    with pytest.raises(KeyboardInterrupt):
        main.ingest_collection(client, alias, "fake/dataset")

    after_collections = {c.name for c in client.get_collections().collections}
    assert before_collections == after_collections, f"orphan left behind: {after_collections - before_collections}"


def test_ingest_collection_count_mismatch_raises_and_cleans_up(memory_client, patch_main_dependencies, monkeypatch):
    '''if uploaded points count does not match expected splits, collection is deleted and error raised.'''
    client = memory_client
    alias = main.alias_for("hf_docs")

    class IncompleteVectorStore:
        def __init__(self, **kwargs):
            pass

        def add_documents(self, docs):
            pass

    monkeypatch.setattr(main, "QdrantVectorStore", IncompleteVectorStore)

    before_collections = {c.name for c in client.get_collections().collections}
    with pytest.raises(RuntimeError, match="expected 8 points"):
        main.ingest_collection(client, alias, "fake/dataset")

    after_collections = {c.name for c in client.get_collections().collections}
    assert before_collections == after_collections


def test_get_or_create_retriever_cold_and_warm(memory_client, patch_main_dependencies):
    '''orchestration builds on cold start, reuses on warm start, sweeps stale builds.'''
    client = memory_client
    alias = main.alias_for("hf_docs")

    # cold start: builds collection and alias
    retriever = main.get_or_create_retriever(client, "hf_docs", "fake/dataset")
    assert client.collection_exists(alias)

    first_target = [a.collection_name for a in client.get_aliases().aliases if a.alias_name == alias][0]
    hits = retriever.invoke("transformers")
    assert len(hits) > 0

    # warm start: should reuse the same build without creating a new one
    retriever2 = main.get_or_create_retriever(client, "hf_docs", "fake/dataset")
    second_target = [a.collection_name for a in client.get_aliases().aliases if a.alias_name == alias][0]
    assert first_target == second_target

    builds = [c.name for c in client.get_collections().collections if c.name.startswith(f"{alias}__build_")]
    assert builds == [first_target]


def test_startup_sweeps_prior_unaliased_orphans(memory_client, patch_main_dependencies):
    '''unaliased build from an unhandled prior crash is swept on next startup before existence check.'''
    client = memory_client
    alias = main.alias_for("hf_docs")

    orphan = f"{alias}__build_20260101T000000Z_deadbeef"
    client.create_collection(orphan, VectorParams(size=8, distance=Distance.COSINE))

    main.get_or_create_retriever(client, "hf_docs", "fake/dataset")

    assert not client.collection_exists(orphan), "orphan build survived cleanup"
    assert client.collection_exists(alias), "alias was not established"


def test_live_schema_migration_via_alias_swap(memory_client, patch_main_dependencies):
    '''simulates a live schema migration (e.g. dense-only to BM25 hybrid/sparse).

    1. Initial schema (Build 1) is live behind alias.
    2. New schema (Build 2) is built in isolation while Build 1 continues serving.
    3. Alias is swapped atomically to Build 2.
    4. Stale Build 1 is cleaned up, completing zero-downtime migration.
    '''
    client = memory_client
    alias = main.alias_for("hf_docs")

    # 1. Build initial index
    retriever_v1 = main.get_or_create_retriever(client, "hf_docs", "fake/dataset")
    build_v1 = [a.collection_name for a in client.get_aliases().aliases if a.alias_name == alias][0]
    assert retriever_v1.invoke("transformers")

    # 2. Build second index (representing schema with BM25 named sparse vectors)
    build_v2 = main.ingest_collection(client, alias, "fake/dataset")
    assert build_v2 != build_v1

    # verify alias still serves build_v1 until swap
    current_target = [a.collection_name for a in client.get_aliases().aliases if a.alias_name == alias][0]
    assert current_target == build_v1

    # 3. Swap alias to build_v2
    main.swap_alias(client, alias, build_v2)
    new_target = [a.collection_name for a in client.get_aliases().aliases if a.alias_name == alias][0]
    assert new_target == build_v2

    # 4. Clean up old build_v1
    main.cleanup_builds(client, alias)
    remaining_collections = sorted(c.name for c in client.get_collections().collections)
    assert remaining_collections == [build_v2]
