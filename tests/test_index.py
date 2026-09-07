"""The five scenarios the alias-swap design was specified against, plus retention."""

import pytest
from qdrant_client.http.models import Distance, VectorParams

import index
from config import IndexConfig
from index import (
    build_index,
    cleanup_builds,
    current_target,
    ensure_index,
    list_builds,
    rollback,
    swap_alias,
)


def test_cold_start_builds_collection_and_creates_alias(client, cfg, embeddings, loader):
    alias = ensure_index(client, cfg, embeddings, loader=loader)

    assert alias == cfg.alias
    assert client.collection_exists(alias)
    target = current_target(client, alias)
    assert target is not None and target.startswith(cfg.build_prefix)
    assert client.count(alias, exact=True).count == 6


def test_warm_start_reuses_alias_without_re_embedding(client, cfg, embeddings, loader):
    ensure_index(client, cfg, embeddings, loader=loader)
    first_target = current_target(client, cfg.alias)

    calls = []

    def counting_loader(config):
        calls.append(config)
        return loader(config)

    ensure_index(client, cfg, embeddings, loader=counting_loader)

    assert calls == [], "warm start must not load or re-embed the dataset"
    assert current_target(client, cfg.alias) == first_target


def test_interrupted_ingest_leaves_no_residue(client, cfg, embeddings, loader):
    def exploding_loader(config):
        docs = loader(config)
        raise KeyboardInterrupt("simulated Ctrl-C mid-ingest")

    with pytest.raises(KeyboardInterrupt):
        ensure_index(client, cfg, embeddings, loader=exploding_loader)

    assert not client.collection_exists(cfg.alias)
    assert list_builds(client, cfg) == []


def test_crashed_build_is_swept_and_rebuilt(client, cfg, embeddings, loader):
    """An orphan build from a hard kill: no alias, so the next startup rebuilds."""
    orphan = f"{cfg.build_prefix}20200101T000000Z_deadbeef"
    client.create_collection(orphan, VectorParams(size=32, distance=Distance.COSINE))
    assert not client.collection_exists(cfg.alias)

    ensure_index(client, cfg, embeddings, loader=loader)

    assert orphan not in list_builds(client, cfg), "orphan should be swept"
    assert client.collection_exists(cfg.alias)


def test_changing_embedding_model_yields_new_alias(client, cfg, embeddings, loader):
    ensure_index(client, cfg, embeddings, loader=loader)

    other = IndexConfig(
        base=cfg.base,
        dataset=cfg.dataset,
        description=cfg.description,
        embedding_model="text-embedding-3-large",
    )
    assert other.alias != cfg.alias

    ensure_index(client, other, embeddings, loader=loader)

    assert client.collection_exists(cfg.alias), "original index must survive untouched"
    assert client.collection_exists(other.alias)


def test_upload_shortfall_discards_the_build(client, cfg, embeddings, loader, monkeypatch):
    """The verification gate: a build whose count is short never becomes live."""
    real_count = client.count

    def short_count(collection_name, **kwargs):
        result = real_count(collection_name, exact=True)
        result.count -= 1
        return result

    monkeypatch.setattr(client, "count", short_count)

    with pytest.raises(RuntimeError, match="expected 6 points"):
        build_index(client, cfg, embeddings, loader=loader)

    assert list_builds(client, cfg) == []
    assert not client.collection_exists(cfg.alias)


def test_retention_keeps_one_rollback_target(client, cfg, embeddings, loader):
    """Three successive builds leave the live one plus exactly one predecessor."""
    for _ in range(3):
        build = build_index(client, cfg, embeddings, loader=loader)
        swap_alias(client, cfg.alias, build)
        cleanup_builds(client, cfg, min_age_minutes=0)

    builds = list_builds(client, cfg)
    assert len(builds) == 2, builds
    assert current_target(client, cfg.alias) == builds[0]


def test_rollback_moves_alias_to_previous_build(client, cfg, embeddings, loader):
    first = build_index(client, cfg, embeddings, loader=loader)
    swap_alias(client, cfg.alias, first)
    second = build_index(client, cfg, embeddings, loader=loader)
    swap_alias(client, cfg.alias, second)

    assert current_target(client, cfg.alias) == second
    assert rollback(client, cfg) == first
    assert current_target(client, cfg.alias) == first


def test_young_builds_are_never_deleted(client, cfg, embeddings, loader):
    """A build with no alias may be another process's upload still in flight."""
    in_flight = build_index(client, cfg, embeddings, loader=loader)
    assert current_target(client, cfg.alias) is None

    deleted = cleanup_builds(client, cfg, min_age_minutes=30)

    assert deleted == []
    assert in_flight in list_builds(client, cfg)