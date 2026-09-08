"""The five scenarios the alias-swap design was specified against, plus retention."""

import dataclasses

import pytest
from qdrant_client.http.models import Distance, VectorParams

from index import (
    build_index,
    cleanup_builds,
    current_target,
    ensure_index,
    list_builds,
    rollback,
    swap_alias,
)


def test_cold_start_builds_collection_and_creates_alias(client, cfg, encoders, loader):
    alias = ensure_index(client, cfg, encoders, loader=loader)

    assert alias == cfg.alias
    assert client.collection_exists(alias)
    target = current_target(client, alias)
    assert target is not None and target.startswith(cfg.build_prefix)
    assert client.count(alias, exact=True).count == 6


def test_warm_start_reuses_alias_without_re_embedding(client, cfg, encoders, loader):
    ensure_index(client, cfg, encoders, loader=loader)
    first_target = current_target(client, cfg.alias)

    calls = []

    def counting_loader(config):
        calls.append(config)
        return loader(config)

    ensure_index(client, cfg, encoders, loader=counting_loader)

    assert calls == [], "warm start must not load or re-embed the dataset"
    assert current_target(client, cfg.alias) == first_target


def test_interrupted_ingest_leaves_no_residue(client, cfg, encoders, loader):
    def exploding_loader(config):
        docs = loader(config)
        raise KeyboardInterrupt("simulated Ctrl-C mid-ingest")

    with pytest.raises(KeyboardInterrupt):
        ensure_index(client, cfg, encoders, loader=exploding_loader)

    assert not client.collection_exists(cfg.alias)
    assert list_builds(client, cfg) == []


def test_crashed_build_is_swept_and_rebuilt(client, cfg, encoders, loader):
    """An orphan build from a hard kill: no alias, so the next startup rebuilds."""
    orphan = f"{cfg.build_prefix}20200101T000000Z_deadbeef"
    client.create_collection(orphan, VectorParams(size=32, distance=Distance.COSINE))
    assert not client.collection_exists(cfg.alias)

    ensure_index(client, cfg, encoders, loader=loader)

    assert orphan not in list_builds(client, cfg), "orphan should be swept"
    assert client.collection_exists(cfg.alias)


def test_changing_embedding_model_yields_new_alias(client, cfg, encoders, loader):
    ensure_index(client, cfg, encoders, loader=loader)

    # the declared model moves on both sides at once: a config naming one model and an
    # encoder naming another is exactly what require_encoders exists to reject.
    other = dataclasses.replace(cfg, embedding_model="fake-dense-other")
    other_encoders = dataclasses.replace(encoders, dense_model="fake-dense-other")
    assert other.alias != cfg.alias

    ensure_index(client, other, other_encoders, loader=loader)

    assert client.collection_exists(cfg.alias), "original index must survive untouched"
    assert client.collection_exists(other.alias)


def test_upload_shortfall_discards_the_build(
    client, cfg, encoders, loader, monkeypatch
):
    """The verification gate: a build whose count is short never becomes live."""
    real_count = client.count

    def short_count(collection_name, **kwargs):
        result = real_count(collection_name, exact=True)
        result.count -= 1
        return result

    monkeypatch.setattr(client, "count", short_count)

    with pytest.raises(RuntimeError, match="expected 6 points"):
        build_index(client, cfg, encoders, loader=loader)

    assert list_builds(client, cfg) == []
    assert not client.collection_exists(cfg.alias)


def test_retention_keeps_one_rollback_target(client, cfg, encoders, loader):
    """Three successive builds leave the live one plus exactly one predecessor."""
    for _ in range(3):
        build = build_index(client, cfg, encoders, loader=loader)
        swap_alias(client, cfg.alias, build)
        cleanup_builds(client, cfg, min_age_minutes=0)

    builds = list_builds(client, cfg)
    assert len(builds) == 2, builds
    assert current_target(client, cfg.alias) == builds[0]


def test_rollback_moves_alias_to_previous_build(client, cfg, encoders, loader):
    first = build_index(client, cfg, encoders, loader=loader)
    swap_alias(client, cfg.alias, first)
    second = build_index(client, cfg, encoders, loader=loader)
    swap_alias(client, cfg.alias, second)

    assert current_target(client, cfg.alias) == second
    assert rollback(client, cfg) == first
    assert current_target(client, cfg.alias) == first


def test_young_builds_are_never_deleted(client, cfg, encoders, loader):
    """A build with no alias may be another process's upload still in flight."""
    in_flight = build_index(client, cfg, encoders, loader=loader)
    assert current_target(client, cfg.alias) is None

    deleted = cleanup_builds(client, cfg, min_age_minutes=30)

    assert deleted == []
    assert in_flight in list_builds(client, cfg)

def test_hybrid_build_configures_idf_modifier(client, cfg, encoders, loader):
    """The IDF trap, guarded.

    Without modifier=IDF the sparse vector scores raw term frequency, which is not BM25
    and fails silently. This is the one line whose absence produces no error anywhere.
    """
    from qdrant_client.http.models import Modifier

    from config import DENSE_VECTOR, SPARSE_VECTOR

    build = build_index(client, cfg, encoders, loader=loader)
    params = client.get_collection(build).config.params

    assert list(params.vectors.keys()) == [DENSE_VECTOR]
    assert params.sparse_vectors[SPARSE_VECTOR].modifier == Modifier.IDF


def test_dense_only_build_has_no_sparse_vector(client, dense_cfg, dense_encoders, loader):
    build = build_index(client, dense_cfg, dense_encoders, loader=loader)
    assert not client.get_collection(build).config.params.sparse_vectors


def test_hybrid_config_without_encoder_fails_before_creating_anything(
    client, cfg, dense_encoders, loader
):
    with pytest.raises(ValueError, match="no sparse encoder"):
        build_index(client, cfg, dense_encoders, loader=loader)
    assert list_builds(client, cfg) == []


def test_warm_start_without_encoder_fails_too(
    client, cfg, encoders, dense_encoders, loader
):
    """The warm-start return skips build_index, so the guard must precede it."""
    ensure_index(client, cfg, encoders, loader=loader)
    target = current_target(client, cfg.alias)

    with pytest.raises(ValueError, match="no sparse encoder"):
        ensure_index(client, cfg, dense_encoders, loader=loader)

    assert current_target(client, cfg.alias) == target, "the live index must be untouched"


def test_encoder_for_a_different_dense_model_is_refused(client, cfg, encoders, loader):
    """The alias spells out embedding_model, so the encoder has to actually be it.

    Nothing downstream would notice: the collection is sized from the encoder it was
    handed, so it stays self-consistent and returns plausible rankings forever.
    """
    wrong = dataclasses.replace(encoders, dense_model="text-embedding-3-large")

    with pytest.raises(ValueError, match="declares embedding_model"):
        build_index(client, cfg, wrong, loader=loader)

    assert list_builds(client, cfg) == []


def test_encoder_for_a_different_sparse_model_is_refused(client, cfg, encoders, loader):
    """Same for sparse, where there isn't even a dimension to disagree about."""
    wrong = dataclasses.replace(encoders, sparse_model="Qdrant/splade")

    with pytest.raises(ValueError, match="declares sparse_model"):
        build_index(client, cfg, wrong, loader=loader)

    assert list_builds(client, cfg) == []


def test_fingerprint_covers_chunking_but_not_description(cfg):
    import dataclasses

    assert dataclasses.replace(cfg, chunk_size=512).alias != cfg.alias
    assert dataclasses.replace(cfg, chunk_overlap=0).alias != cfg.alias
    assert dataclasses.replace(cfg, sparse_model=None).alias != cfg.alias
    assert dataclasses.replace(cfg, description="reworded").alias == cfg.alias