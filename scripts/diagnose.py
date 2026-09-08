# dense vs sparse, side by side. does hybrid earn its keep?

from __future__ import annotations

import sys

from dotenv import load_dotenv

from config import CORPORA, require_env
from index import Encoders, get_client
from retrieval import Passage, KnowledgeBase, reciprocal_rank_fusion

COLUMN = 54

# a rare literal token is BM25's home turf: one exact match, no paraphrase to bridge.
# a conceptual phrase is the dense index's, since none of its words need to appear.
# if the two columns look the same on both, fusion is buying nothing.
DEFAULT_QUERIES = [
    ("AutoModelForCausalLM", "rare literal token"),
    ("how do I stop the model from repeating itself", "conceptual phrase"),
]

def label(passage: Passage) -> str:
    """origin plus the head of the chunk, collapsed onto one line."""
    text = " ".join(passage.text.split())
    return f"{passage.origin} | {text}"

def cell(passage: Passage | None, shared: set[str]) -> str:
    if passage is None:
        return " " * COLUMN
    mark = "*" if passage.id in shared else " "
    return f"{mark} {label(passage)}"[:COLUMN].ljust(COLUMN)

def side_by_side(dense: list[Passage], sparse: list[Passage], k: int, hybrid: bool) -> None:
    shared = {p.id for p in dense} & {p.id for p in sparse}

    print(f"  {'#':<3} {'DENSE':<{COLUMN}} SPARSE")
    print(f"  {'-' * 3} {'-' * COLUMN} {'-' * COLUMN}")
    for rank in range(k):
        d = dense[rank] if rank < len(dense) else None
        s = sparse[rank] if rank < len(sparse) else None
        if d is None and s is None:
            break
        print(f"  {rank + 1:<3} {cell(d, shared)} {cell(s, shared)}")

    if sparse:
        print(f"\n  * = also in the other column. overlap {len(shared)}/{max(len(dense), len(sparse))}.")
    elif not hybrid:
        print("\n  (corpus is dense-only: there is no sparse index to query)")
    else:
        # an empty sparse ranking is a result, not an error: BM25 only returns points
        # sharing a term with the query, so zero hits means the corpus lacks the term.
        print("\n  (sparse returned 0 hits: no chunk here contains any query term.")
        print("   on a rare-literal probe that is the finding, not a failure.)")

def fused_top(dense: list[Passage], sparse: list[Passage], n: int = 5) -> None:
    """what retrieve() would actually hand the model, and which retrievers found it."""
    rankings = [r for r in (dense, sparse) if r]
    print("\n  after RRF:")
    for rank, passage in enumerate(reciprocal_rank_fusion(rankings)[:n], start=1):
        found = "+".join(
            name
            for name, value in (("dense", passage.dense_rank), ("sparse", passage.sparse_rank))
            if value is not None
        )
        print(f"  {rank:<3} [{found:<12}] {label(passage)[:COLUMN * 2 - 18]}")

def report(kb: KnowledgeBase, query: str, kind: str, k: int) -> None:
    for cfg in kb.configs:
        print(f"\n{'=' * (COLUMN * 2 + 7)}")
        print(f"QUERY  {query!r}")
        print(f"       {kind or 'query'} · corpus={cfg.base}")
        print(f"{'=' * (COLUMN * 2 + 7)}")

        dense = kb.dense_rankings(query, limit=k, sources=[cfg.base])
        sparse = kb.sparse_rankings(query, limit=k, sources=[cfg.base])
        dense = dense[0] if dense else []
        sparse = sparse[0] if sparse else []

        side_by_side(dense, sparse, k, cfg.hybrid)
        fused_top(dense, sparse)

def main() -> None:
    load_dotenv()
    require_env("QDRANT_URL", "QDRANT_KEY", "OPENAI_API_KEY")

    given = sys.argv[1:]
    queries = [(q, "") for q in given] if given else DEFAULT_QUERIES

    encoders = Encoders.for_config(CORPORA[0])
    kb = KnowledgeBase(get_client(), CORPORA, encoders).ensure_indexes()

    for query, kind in queries:
        report(kb, query, kind, k=10)

if __name__ == "__main__":
    main()
