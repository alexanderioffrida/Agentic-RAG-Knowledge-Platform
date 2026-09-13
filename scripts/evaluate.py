# does hybrid earn its keep? recall@k and MRR over the golden set.

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from config import CANDIDATE_LIMIT, CORPORA, RERANK_DEPTH, require_env
from goldens import Golden, GOLDENS_PATH, load_goldens, stale_sources
from index import Encoders, Reranker, get_client
from retrieval import KnowledgeBase, Passage, reciprocal_rank_fusion, rerank

RESULTS_DIR = Path(__file__).resolve().parents[1] / "evals" / "results"
KS = (1, 5, 20)

# every arm fuses with the same RRF over the same per-corpus lists, so the only thing
# that varies between them is which retrievers feed the pool. an arm that merged corpora
# differently would be measuring two changes at once. `reranked` extends that discipline
# one step further: it is hybrid's pool reordered, never a different pool.
ARMS = ("dense", "sparse", "hybrid", "reranked")

def arms_in(rows: list[dict]) -> tuple[str, ...]:
    """the arms actually scored, in ARMS order.

    `reranked` is absent when no reranker was supplied, which is what keeps the
    three-arm baseline reproducible after the cross-encoder lands.
    """
    present = set(rows[0]["ranks"]) if rows else set()
    return tuple(arm for arm in ARMS if arm in present)

def rankings_for(kb: KnowledgeBase, query: str, limit: int) -> dict[str, list[Passage]]:
    """all three arms off one dense embedding and one sparse embedding.

    running the arms independently would re-embed the query three times and, for dense,
    pay three OpenAI round trips per question. the arms are views on the same two calls.
    """
    dense = kb.dense_rankings(query, limit=limit)
    sparse = kb.sparse_rankings(query, limit=limit)
    hybrid = reciprocal_rank_fusion(dense + sparse)
    arms = {
        "dense": reciprocal_rank_fusion(dense),
        "sparse": reciprocal_rank_fusion(sparse),
        "hybrid": hybrid
    }
    if kb.reranker is not None:
        # k is the whole pool, not RETRIEVAL_K: the arm has to stay rankable to the same
        # depth as hybrid or every comparison between them measures the cutoff instead.
        arms["reranked"] = rerank(kb.reranker, query, hybrid, k=len(hybrid))
    return arms

def rank_of(passages: list[Passage], chunk: str) -> int | None:
    """1-based position of the gold chunk, or None if it never appears."""
    for rank, passage in enumerate(passages, start=1):
        if passage.key == chunk:
            return rank
    return None

def found_by(passages: list[Passage], chunk: str) -> str:
    """which retrievers turned the gold chunk up. free, because Passage carries both."""
    for passage in passages:
        if passage.key == chunk:
            return "+".join(
                name
                for name, rank in (
                    ("dense", passage.dense_rank), ("sparse", passage.sparse_rank)
                )
                if rank is not None
            )
    return ""

def summarize(ranks: list[int | None], ks: tuple[int, ...] = KS) -> dict:
    """recall@k for each k, plus MRR over the full candidate depth.

    recall@k here is hit rate: one gold chunk per question, so "did it come back" is
    the whole of recall. MRR is the same evidence weighted by how far down it was —
    a change that moves gold from rank 9 to rank 2 shows up in MRR and in nothing else.
    """
    total = len(ranks) or 1
    found = [rank for rank in ranks if rank is not None]
    return {
        "n": len(ranks),
        **{f"recall@{k}": sum(1 for r in found if r <= k) / total for k in ks},
        "mrr": sum(1.0 / r for r in found) / total
    }

def rescued(hybrid: list[int | None], dense: list[int | None], k: int) -> int:
    """questions where the gold chunk is in hybrid's top-k and not in dense's.

    this is the number that justifies the sparse index. an aggregate recall gain of two
    points could be one arm helping some queries and hurting others; this counts the
    queries that were actually saved, and `regressed` counts the ones that were lost.
    """
    return sum(
        1
        for h, d in zip(hybrid, dense)
        if (h is not None and h <= k) and not (d is not None and d <= k)
    )

def evaluate(kb: KnowledgeBase, goldens: list[Golden], limit: int) -> dict:
    """one pass over the golden set, scoring every arm on every question."""
    rows = []
    for i, golden in enumerate(goldens, start=1):
        arms = rankings_for(kb, golden.question, limit)
        rows.append({
            "question": golden.question,
            "style": golden.style,
            "source": golden.source,
            "chunk": golden.chunk,
            "ranks": {arm: rank_of(arms[arm], golden.chunk) for arm in arms},
            "found_by": found_by(arms["hybrid"], golden.chunk)
        })
        if i % 10 == 0:
            print(f"   {i}/{len(goldens)}")
    return {"rows": rows}

def table(rows: list[dict], styles: list[str]) -> str:
    """arms down, k across, one block per style plus the pooled numbers."""
    lines = []
    header = f"  {'arm':<8} {'n':>4} " + " ".join(f"{'recall@' + str(k):>10}" for k in KS) + f" {'MRR':>8}"
    for style in [*styles, "all"]:
        subset = [r for r in rows if style == "all" or r["style"] == style]
        lines += [f"\n{style.upper()}", header, "  " + "-" * (len(header) - 2)]
        for arm in arms_in(rows):
            stats = summarize([r["ranks"][arm] for r in subset])
            cells = " ".join(f"{stats[f'recall@{k}']:>10.2f}" for k in KS)
            lines.append(f"  {arm:<8} {stats['n']:>4} {cells} {stats['mrr']:>8.3f}")
    return "\n".join(lines)

def attribution(rows: list[dict], k: int = 5) -> str:
    """what each stage won and lost against the one before it, question by question."""
    arms = arms_in(rows)
    hybrid = [r["ranks"]["hybrid"] for r in rows]
    dense = [r["ranks"]["dense"] for r in rows]
    hits = [r["found_by"] for r in rows if r["ranks"]["hybrid"] is not None]
    counts = {name: hits.count(name) for name in ("dense", "sparse", "dense+sparse")}

    lines = [
        f"\nAT k={k}",
        f"  rescued by sparse   {rescued(hybrid, dense, k):>4}  "
        "(gold in hybrid top-k, absent from dense top-k)",
        f"  lost by fusion      {rescued(dense, hybrid, k):>4}  "
        "(the reverse — RRF pushing a dense win out of the window)"
    ]

    # the reranker's own ledger, against the pool it was handed. an aggregate recall gain
    # cannot distinguish a clean win from a wash, and the cross-encoder gets no free pass
    # that fusion did not get.
    if "reranked" in arms:
        reranked = [r["ranks"]["reranked"] for r in rows]
        lines += [
            f"  rescued by rerank   {rescued(reranked, hybrid, k):>4}  "
            "(gold in reranked top-k, absent from hybrid top-k)",
            f"  lost by rerank      {rescued(hybrid, reranked, k):>4}  "
            "(the cross-encoder demoting a chunk fusion already had right)"
        ]

    never = [r for r in rows if all(r["ranks"][arm] is None for arm in arms)]
    lines += [
        "\nWHO FOUND THE GOLD CHUNK (anywhere in the pool)",
        *[f"  {name:<18} {count:>4}" for name, count in counts.items()],
        f"\n  found by no arm at any depth: {len(never)}",
        "  (a question no retriever can answer is usually a bad question or a chunk "
        "that left the index — read them before blaming retrieval.)"
        if never else ""
    ]
    return "\n".join(lines)

def main() -> None:
    load_dotenv()
    require_env("QDRANT_URL", "QDRANT_KEY", "OPENAI_API_KEY")

    goldens = load_goldens()
    stale = stale_sources(goldens, CORPORA)
    if stale:
        raise SystemExit(
            f"golden set was written against different chunking for {sorted(stale)}. "
            f"the answer keys are chunk hashes, so a resplit invalidates them: "
            f"rebuild the set with `python scripts/goldens.py`."
        )

    encoders = Encoders.for_config(CORPORA[0])
    reranker = Reranker.for_model()
    kb = KnowledgeBase(
        get_client(), CORPORA, encoders, reranker=reranker
    ).ensure_indexes()

    print(
        f"-> scoring {len(goldens)} questions, candidate depth {CANDIDATE_LIMIT}, "
        f"reranking the top {RERANK_DEPTH} with '{reranker.model}'"
    )
    result = evaluate(kb, goldens, CANDIDATE_LIMIT)
    rows = result["rows"]
    styles = sorted({r["style"] for r in rows})
    arms = arms_in(rows)

    print(table(rows, styles))
    print(attribution(rows))

    # the index every number was measured against, recorded with the numbers. a metric
    # that cannot name its index is not evidence of anything.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{stamp}.json"
    path.write_text(json.dumps({
        "stamp": stamp,
        "goldens": str(GOLDENS_PATH),
        "candidate_limit": CANDIDATE_LIMIT,
        "rerank_depth": RERANK_DEPTH,
        "rerank_model": reranker.model,
        "aliases": {cfg.base: cfg.alias for cfg in CORPORA},
        "summary": {
            style: {
                arm: summarize([
                    r["ranks"][arm] for r in rows
                    if style == "all" or r["style"] == style
                ])
                for arm in arms
            }
            for style in [*styles, "all"]
        },
        "rows": rows
    }, indent=2))
    print(f"\n-> {path}")

if __name__ == "__main__":
    main()
