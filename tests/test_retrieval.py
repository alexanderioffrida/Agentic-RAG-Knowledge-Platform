"""Fusion is pure and cheap to test, so it gets tested directly."""

import dataclasses

import pytest

from retrieval import Passage, reciprocal_rank_fusion


def p(pid, source="a", dense_rank=None, sparse_rank=None):
    return Passage(id=pid, text=pid, source=source,
                   dense_rank=dense_rank, sparse_rank=sparse_rank)


def test_single_ranking_preserves_order():
    ranking = [p("a", dense_rank=1), p("b", dense_rank=2), p("c", dense_rank=3)]
    assert [x.id for x in reciprocal_rank_fusion([ranking])] == ["a", "b", "c"]


def test_agreement_beats_depth():
    """The defining property of RRF: appearing in both lists outweighs topping one."""
    dense = [p("solo", dense_rank=1), p("shared", dense_rank=2)]
    sparse = [p("other", sparse_rank=1), p("shared", sparse_rank=2)]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert fused[0].id == "shared"


def test_fusion_merges_provenance_ranks():
    dense = [p("shared", dense_rank=1)]
    sparse = [p("shared", sparse_rank=4)]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert len(fused) == 1
    assert fused[0].dense_rank == 1 and fused[0].sparse_rank == 4


def test_scores_follow_the_rrf_formula():
    fused = reciprocal_rank_fusion([[p("a", dense_rank=1)]], k=60)
    assert fused[0].fused_score == 1.0 / 61


def test_fusion_does_not_mutate_inputs():
    ranking = [p("a", dense_rank=1)]
    reciprocal_rank_fusion([ranking])
    assert ranking[0].fused_score is None


def test_retrieve_returns_fused_passages_across_corpora(client, dense_encoders, loader):
    """End-to-end: build two indexes, query, get merged Passages with payload intact."""
    kb = _kb(client, dense_encoders, loader)

    rankings = kb.dense_rankings("transformers pipelines", limit=4)
    assert len(rankings) == 2, "one ranked list per corpus"
    assert {r[0].source for r in rankings} == {"alpha", "beta"}

    result = kb.retrieve("transformers pipelines", k=5)

    assert len(result.passages) == 5
    assert all(p.text for p in result.passages), "payload text must survive"
    assert all(p.metadata.get("source") for p in result.passages), "metadata must survive"
    assert all(p.fused_score is not None for p in result.passages)
    scores = [p.fused_score for p in result.passages]
    assert scores == sorted(scores, reverse=True)
    assert result.confidence is None and result.used_web is False

def test_k_controls_how_much_depth_matters():
    """Small k rewards depth; large k rewards agreement. Both are one parameter."""
    dense = [p("solo", dense_rank=1), p("shared", dense_rank=2)]
    sparse = [p("other", sparse_rank=1), p("shared", sparse_rank=2)]

    assert reciprocal_rank_fusion([dense, sparse], k=0)[0].id == "solo"
    assert reciprocal_rank_fusion([dense, sparse], k=60)[0].id == "shared"


def _corpora(encoders):
    """two corpora declaring whatever the given encoders claim to be."""
    from config import IndexConfig

    return [
        IndexConfig(base="alpha", dataset="fake/a", description="a",
                    embedding_model=encoders.dense_model,
                    sparse_model=encoders.sparse_model),
        IndexConfig(base="beta", dataset="fake/b", description="b",
                    embedding_model=encoders.dense_model,
                    sparse_model=encoders.sparse_model),
    ]


def _kb(client, encoders, loader, configs=None):
    from retrieval import KnowledgeBase

    configs = configs or _corpora(encoders)
    return KnowledgeBase(client, configs, encoders).ensure_indexes(loader=loader)


def test_hybrid_retrieve_fuses_four_rankings(client, encoders, loader):
    """Two corpora x two retrievers = four ranked lists into one pool."""
    kb = _kb(client, encoders, loader)

    dense = kb.dense_rankings("transformers pipelines", limit=4)
    sparse = kb.sparse_rankings("transformers pipelines", limit=4)

    assert len(dense) == 2 and len(sparse) == 2
    assert {r[0].source for r in dense} == {"alpha", "beta"}
    assert all(passage.dense_rank is not None for r in dense for passage in r)
    assert all(passage.sparse_rank is not None for r in sparse for passage in r)

    result = kb.retrieve("transformers pipelines", k=5)

    assert len(result.passages) == 5
    assert all(p.text for p in result.passages), "payload text must survive"
    assert all(p.metadata.get("source") for p in result.passages), "metadata must survive"
    scores = [p.fused_score for p in result.passages]
    assert scores == sorted(scores, reverse=True)
    assert result.confidence is None and result.used_web is False


def test_chunks_found_by_both_retrievers_carry_both_ranks(client, encoders, loader):
    kb = _kb(client, encoders, loader)
    result = kb.retrieve("chunk 3 about transformers and pipelines", k=10)

    both = [p for p in result.passages if p.dense_rank and p.sparse_rank]
    assert both, "at least one chunk should surface in both rankings"


def test_dense_only_config_skips_sparse_entirely(client, dense_encoders, loader):
    kb = _kb(client, dense_encoders, loader)

    assert kb.sparse_rankings("anything") == []
    assert len(kb.retrieve("transformers", k=3).passages) == 3


def test_sources_filter_restricts_both_retrievers(client, encoders, loader):
    kb = _kb(client, encoders, loader)

    result = kb.retrieve("transformers", k=6, sources=["alpha"])

    assert {p.source for p in result.passages} == {"alpha"}


def test_knowledge_base_refuses_a_corpus_its_encoders_do_not_match(client, encoders):
    """Validation at construction, not at first query.

    Under the FastAPI layer the encoders get built once in a lifespan handler, so this
    is the difference between a boot that fails and wrong answers served all day.
    """
    from retrieval import KnowledgeBase

    alpha, beta = _corpora(encoders)
    divergent = dataclasses.replace(beta, embedding_model="text-embedding-3-large")

    with pytest.raises(ValueError, match="declares embedding_model"):
        KnowledgeBase(client, [alpha, divergent], encoders)