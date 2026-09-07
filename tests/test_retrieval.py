"""Fusion is pure and cheap to test, so it gets tested directly."""

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


def test_retrieve_returns_fused_passages_across_corpora(client, embeddings, loader):
    """End-to-end: build two indexes, query, get merged Passages with payload intact."""
    from config import IndexConfig
    from retrieval import KnowledgeBase

    configs = [
        IndexConfig(base="alpha", dataset="fake/a", description="a"),
        IndexConfig(base="beta", dataset="fake/b", description="b"),
    ]
    kb = KnowledgeBase(client, configs, embeddings).ensure_indexes(loader=loader)

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