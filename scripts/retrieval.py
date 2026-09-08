# dense • sparse • fuse • rerank • retrieve()

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace

from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import (
    CANDIDATE_LIMIT, DENSE_VECTOR, RETRIEVAL_K, RRF_K, SPARSE_VECTOR, IndexConfig
)
from index import Encoders, ensure_index, require_encoders

# langchain-qdrant's payload convention, so passages can be read straight off a point.
CONTENT_KEY = "page_content"
METADATA_KEY = "metadata"

@dataclass
class Passage:
    """one retrieved chunk, carrying every score it picks up on the way through, a single obj that accumulates scores stage by stage."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    source: str = ""
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None

    @property
    def origin(self) -> str:
        """best available human-facing pointer at where this chunk came from."""
        for key in ("source", "url", "path", "filename"):
            value = self.metadata.get(key)
            if value:
                return str(value)
        return self.source

@dataclass
class RetrievalResult:
    """what retrieval knows, not just what it found.
    
    `confidence` and `escalate` are computed here because this is where the scores live;
    returning a bare list would throw them away and force the API layer to recompute.
    """
    query: str
    passages: list[Passage]
    confidence: float | None = None
    used_web: bool = False
    escalate: bool = False

def passage_from_point(point, source: str, rank: int, kind: str="dense") -> Passage:
    payload = point.payload or {}
    return Passage(
        id=str(point.id),
        text=payload.get(CONTENT_KEY, ""),
        metadata=dict(payload.get(METADATA_KEY) or {}),
        source=source,
        dense_rank=rank if kind == "dense" else None,
        sparse_rank=rank if kind == "sparse" else None
    )

def reciprocal_rank_fusion(
    rankings: list[list[Passage]], k: int = RRF_K
) -> list[Passage]:
    """merges ranked lists by summing 1 / (k + rank), highest total first.
    
    ranks only, never the underlying scores: cosine similarity and BM25 are on
    incomparable scales, and any normalization between them is arbitrary and unstable
    across queries. rank is scale-free, so fusion needs no calibration.
    """
    merged: dict[str, Passage] = {}
    scores: dict[str, float] = defaultdict(float)

    for ranking in rankings:
        for rank, passage in enumerate(ranking, start=1):
            scores[passage.id] += 1.0 / (k + rank)
            existing = merged.get(passage.id)
            if existing is None:
                merged[passage.id] = replace(passage)
                continue
            if passage.dense_rank is not None:
                existing.dense_rank = passage.dense_rank
            if passage.sparse_rank is not None:
                existing.sparse_rank = passage.sparse_rank
    
    for point_id, passage in merged.items():
        passage.fused_score = scores[point_id]
    
    return sorted(merged.values(), key=lambda p: p.fused_score or 0.0, reverse=True)

def format_for_llm(passages: list[Passage]) -> str:
    """renders passages for the model. metadata still travels separately as the artifact."""
    if not passages:
        return "No relevant passages found in the indexed documentation."
    blocks = [
        f"[{i}] corpus={p.source} origin={p.origin}\n{p.text}"
        for i, p in enumerate(passages, start=1)
    ]
    return "\n\n".join(blocks)

class KnowledgeBase:
    """owns the indexes and the retrieval pipeline over them."""
    def __init__(
        self,
        client: QdrantClient,
        configs: list[IndexConfig] | tuple[IndexConfig, ...],
        encoders: Encoders
    ) -> None:
        # one set of encoders serves every corpus, because retrieve() fuses them into a
        # single pool and rankings from two embedding spaces are not comparable. so every
        # config has to declare these models. checked here, at startup, rather than at
        # first query: under the service layer that is the difference between a failed
        # boot and a wrong answer served for the life of the process.
        for cfg in configs:
            require_encoders(cfg, encoders)

        self.client = client
        self.configs = configs
        self.encoders = encoders
        self._aliases: dict[str, str] = {}

    def ensure_indexes(self, loader=None) -> "KnowledgeBase":
        for cfg in self.configs:
            self._aliases[cfg.base] = ensure_index(
                self.client, cfg, self.encoders, loader=loader
            )
        return self

    @property
    def sources(self) -> list[str]:
        return [cfg.base for cfg in self.configs]

    def _selected(self, sources: list[str] | None) -> list[IndexConfig]:
        if not sources:
            return self.configs
        wanted = set(sources)
        return [cfg for cfg in self.configs if cfg.base in wanted]

    def dense_rankings(
        self,
        query: str,
        limit: int = CANDIDATE_LIMIT,
        sources: list[str] | None = None
    ) -> list[list[Passage]]:
        """one ranked list per corpus. fusion merges them; nothing chooses between them."""
        vector = self.encoders.dense.embed_query(query)
        return [
            self._search(cfg, vector, DENSE_VECTOR, limit, "dense")
            for cfg in self._selected(sources)
        ]

    def sparse_rankings(
        self,
        query: str,
        limit: int = CANDIDATE_LIMIT,
        sources: list[str] | None = None
    ) -> list[list[Passage]]:
        """one ranked list per corpus. fusion merges them; nothing chooses between them."""
        if self.encoders.sparse is None:
            return []
        sparse = self.encoders.sparse.embed_query(query)
        vector = models.SparseVector(indices=sparse.indices, values=sparse.values)
        return [
            self._search(cfg, vector, SPARSE_VECTOR, limit, "sparse")
            for cfg in self._selected(sources)
            if cfg.hybrid
        ]

    def _search(self, cfg, query_vector, using: str, limit: int, kind: str):
        points = self.client.query_points(
            collection_name=self._aliases[cfg.base],
            query=query_vector,
            using=using,
            limit=limit,
            with_payload=True
        ).points
        return [
            passage_from_point(point, cfg.base, rank, kind=kind)
            for rank, point in enumerate(points, start=1)
        ]

    def retrieve(
        self,
        query: str,
        k: int = RETRIEVAL_K,
        sources: list[str] | None = None
    ) -> RetrievalResult:
        """the single entry point. the agent, the api, and the eval harness all call this."""
        rankings = self.dense_rankings(query, sources=sources)
        rankings += self.sparse_rankings(query, sources=sources)
        fused = reciprocal_rank_fusion(rankings)
        # confidence stays None until a cross-encoder is in the path
        # RRF scores are a narrow function of rank and say nothing about abs relevance,
        # so thresholding on them would be meaningless.
        return RetrievalResult(query=query, passages=fused[:k])

