"""The metrics are pure, so they get tested directly — and the harness end to end."""

import dataclasses

import pytest

from config import IndexConfig
from evaluate import (
    arms_in,
    attribution,
    evaluate,
    found_by,
    rank_of,
    rankings_for,
    rescued,
    summarize,
)
from goldens import Golden, load_goldens, stale_sources, write_goldens
from retrieval import KnowledgeBase, Passage, chunk_key


def p(text, dense_rank=None, sparse_rank=None):
    return Passage(id=text, text=text, dense_rank=dense_rank, sparse_rank=sparse_rank)


def test_rank_is_one_based_and_none_when_absent():
    passages = [p("a"), p("b"), p("c")]
    assert rank_of(passages, chunk_key("a")) == 1
    assert rank_of(passages, chunk_key("c")) == 3
    assert rank_of(passages, chunk_key("z")) is None


def test_recall_counts_hits_inside_k_and_mrr_weights_depth():
    stats = summarize([1, 3, None, 20], ks=(1, 5))
    assert stats["n"] == 4
    assert stats["recall@1"] == 0.25
    assert stats["recall@5"] == 0.5
    # 1/1 + 1/3 + 0 + 1/20, over four questions
    assert stats["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 20) / 4)


def test_a_miss_is_worth_zero_not_dropped():
    """averaging over found-only would make a retriever look better the less it finds."""
    assert summarize([1, None])["mrr"] == pytest.approx(0.5)
    assert summarize([1])["mrr"] == pytest.approx(1.0)


def test_rescue_counts_queries_not_score_differences():
    hybrid = [1, 9, None, 2]
    dense = [None, 2, None, 3]
    # only question 0: inside hybrid's top-5 and absent from dense's
    assert rescued(hybrid, dense, k=5) == 1
    # question 1 goes the other way: dense had it at 2, fusion pushed it to 9
    assert rescued(dense, hybrid, k=5) == 1


def test_found_by_reads_provenance_off_the_passage():
    pool = [p("a", dense_rank=4, sparse_rank=1), p("b", dense_rank=2)]
    assert found_by(pool, chunk_key("a")) == "dense+sparse"
    assert found_by(pool, chunk_key("b")) == "dense"
    assert found_by(pool, chunk_key("z")) == ""


def test_stale_sources_ignores_n_docs_but_catches_a_resplit():
    cfg = IndexConfig(base="docs", dataset="d", description="x")
    golden = Golden(
        question="q", style="literal", chunk="abc", source="docs",
        chunk_fingerprint=cfg.chunk_fingerprint, head="",
    )

    bigger = dataclasses.replace(cfg, n_docs=cfg.n_docs * 10)
    assert not stale_sources([golden], [bigger]), "more documents, same chunk text"

    resplit = dataclasses.replace(cfg, chunk_size=cfg.chunk_size // 2)
    assert stale_sources([golden], [resplit]), "a resplit invalidates every answer key"


def test_goldens_round_trip(tmp_path):
    golden = Golden(
        question="how do I load a tokenizer?", style="conceptual", chunk="deadbeef",
        source="docs", chunk_fingerprint="cafe1234", head="AutoTokenizer...",
    )
    path = write_goldens([golden], tmp_path / "goldens.jsonl")
    assert load_goldens(path) == [golden]


def test_missing_golden_file_says_how_to_build_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/goldens.py"):
        load_goldens(tmp_path / "absent.jsonl")


# --- end to end, against a real in-memory Qdrant -----------------------------------

def _corpus(base, encoders):
    return IndexConfig(
        base=base, dataset="fake/dataset", description=base,
        embedding_model=encoders.dense_model, sparse_model=encoders.sparse_model,
    )


@pytest.fixture
def kb(client, encoders, loader):
    return KnowledgeBase(
        client, [_corpus("alpha", encoders)], encoders
    ).ensure_indexes(loader=loader)


def test_a_question_quoting_its_chunk_is_found_by_every_arm(kb, loader):
    """The harness's own smoke test: if a verbatim query misses, the harness is wrong.

    The loader's chunks are distinct, so the chunk quoted back at the index is the one
    that should come first — under dense similarity, under BM25, and under fusion.
    """
    gold = loader(None)[3].page_content
    golden = Golden(
        question=gold, style="literal", chunk=chunk_key(gold), source="alpha",
        chunk_fingerprint="unused-here", head=gold[:40],
    )

    rows = evaluate(kb, [golden], limit=10)["rows"]
    assert rows[0]["ranks"] == {"dense": 1, "sparse": 1, "hybrid": 1}
    assert rows[0]["found_by"] == "dense+sparse"


def test_a_question_about_nothing_in_the_corpus_misses_every_arm(kb):
    golden = Golden(
        question="quarterly revenue for the Belgian subsidiary", style="conceptual",
        chunk=chunk_key("a chunk that was never indexed"), source="alpha",
        chunk_fingerprint="unused-here", head="",
    )

    rows = evaluate(kb, [golden], limit=10)["rows"]
    assert rows[0]["ranks"] == {"dense": None, "sparse": None, "hybrid": None}
    assert "found by no arm at any depth: 1" in attribution(rows)


class Counting:
    """counts embed_query, delegates everything the rankings actually use."""

    def __init__(self, inner, counter, name):
        self.inner, self.counter, self.name = inner, counter, name

    def embed_query(self, text):
        self.counter[self.name] += 1
        return self.inner.embed_query(text)


def test_arms_share_one_embedding_pass(kb):
    """Three arms, one dense call and one sparse call. The reason they are views."""
    calls = {"dense": 0, "sparse": 0}
    kb.encoders = dataclasses.replace(
        kb.encoders,
        dense=Counting(kb.encoders.dense, calls, "dense"),
        sparse=Counting(kb.encoders.sparse, calls, "sparse"),
    )

    arms = rankings_for(kb, "transformers pipelines", limit=5)
    assert set(arms) == {"dense", "sparse", "hybrid"}
    assert calls == {"dense": 1, "sparse": 1}


def test_arms_in_omits_reranked_until_it_is_scored():
    """the three-arm baseline stays printable after the fourth arm lands."""
    three = [{"ranks": {"dense": 1, "sparse": 2, "hybrid": 1}}]
    four = [{"ranks": {"dense": 1, "sparse": 2, "hybrid": 1, "reranked": 3}}]
    assert arms_in(three) == ("dense", "sparse", "hybrid")
    assert arms_in(four) == ("dense", "sparse", "hybrid", "reranked")


def test_reranked_arm_is_hybrid_reordered_not_a_new_pool(kb, reranker):
    """same passages, same length — only the order may change.

    if this arm were truncated to RETRIEVAL_K, every rank comparison with hybrid
    would measure the cutoff rather than the model.
    """
    kb.reranker = reranker
    arms = rankings_for(kb, "transformers pipelines", limit=5)

    assert set(arms) == {"dense", "sparse", "hybrid", "reranked"}
    assert len(arms["reranked"]) == len(arms["hybrid"])
    assert {p.id for p in arms["reranked"]} == {p.id for p in arms["hybrid"]}


def test_reranked_arm_does_not_re_embed(kb, reranker):
    calls = {"dense": 0, "sparse": 0}
    kb.reranker = reranker
    kb.encoders = dataclasses.replace(
        kb.encoders,
        dense=Counting(kb.encoders.dense, calls, "dense"),
        sparse=Counting(kb.encoders.sparse, calls, "sparse"),
    )

    rankings_for(kb, "transformers pipelines", limit=5)
    assert calls == {"dense": 1, "sparse": 1}


def test_evaluate_scores_the_reranked_arm(kb, reranker, loader):
    kb.reranker = reranker
    gold = loader(None)[3].page_content
    golden = Golden(
        question=gold, style="literal", chunk=chunk_key(gold), source="alpha",
        chunk_fingerprint="unused-here", head=gold[:40],
    )

    rows = evaluate(kb, [golden], limit=10)["rows"]
    assert set(rows[0]["ranks"]) == {"dense", "sparse", "hybrid", "reranked"}
    assert rows[0]["ranks"]["reranked"] == 1


def test_attribution_reports_rerank_rescues_and_losses():
    rows = [
        {
            "ranks": {"dense": 1, "sparse": 1, "hybrid": 6, "reranked": 2},
            "found_by": "dense+sparse",
        },
        {
            "ranks": {"dense": 1, "sparse": 1, "hybrid": 2, "reranked": 9},
            "found_by": "dense+sparse",
        },
    ]
    text = attribution(rows, k=5)
    assert "rescued by rerank" in text and "lost by rerank" in text
    # question 0: hybrid 6 (out) -> reranked 2 (in); question 1 is the reverse
    assert "rescued by rerank      1" in text
    assert "lost by rerank         1" in text
