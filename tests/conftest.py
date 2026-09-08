import pytest
import hashlib
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector
from qdrant_client import QdrantClient

from config import IndexConfig
from index import Encoders

DIMENSION = 32

# the fakes name themselves. a config that declares these is telling the truth, and the
# alias reads test_docs__fake_dense_32__n50__... instead of claiming an OpenAI model it
# has never called.
FAKE_DENSE_MODEL = "fake-dense-32"
FAKE_SPARSE_MODEL = "fake-bm25"


@pytest.fixture
def client():
    """A real Qdrant, in process. No cloud calls, no embedding spend."""
    c = QdrantClient(":memory:")
    yield c
    c.close()


@pytest.fixture
def embeddings():
    return DeterministicFakeEmbedding(size=DIMENSION)


@pytest.fixture
def cfg():
    return IndexConfig(
        base="test_docs", dataset="fake/dataset", description="fixture corpus",
        embedding_model=FAKE_DENSE_MODEL, sparse_model=FAKE_SPARSE_MODEL,
    )


@pytest.fixture
def loader():
    def _load(cfg):
        return [
            Document(page_content=f"chunk {i} about transformers and pipelines",
                     metadata={"source": f"doc-{i}"})
            for i in range(6)
        ]
    return _load


class FakeSparseEmbedding(SparseEmbeddings):
    """A deterministic stand-in for fastembed's BM25 encoder.

    Same contract: term frequencies on the document side, flat weights on the query
    side, and no view of the corpus. Keeps the suite offline and model-download free.
    """

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [t for t in text.lower().split() if t]

    @staticmethod
    def _index(token: str) -> int:
        return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        vectors = []
        for text in texts:
            counts: dict[int, float] = {}
            for token in self._tokens(text):
                counts[self._index(token)] = counts.get(self._index(token), 0.0) + 1.0
            items = sorted(counts.items())
            vectors.append(
                SparseVector(indices=[i for i, _ in items], values=[v for _, v in items])
            )
        return vectors

    def embed_query(self, text: str) -> SparseVector:
        indices = sorted({self._index(t) for t in self._tokens(text)})
        return SparseVector(indices=indices, values=[1.0] * len(indices))


@pytest.fixture
def sparse_embeddings():
    return FakeSparseEmbedding()


@pytest.fixture
def dense_cfg():
    return IndexConfig(
        base="test_docs", dataset="fake/dataset", description="dense only",
        embedding_model=FAKE_DENSE_MODEL, sparse_model=None,
    )


@pytest.fixture
def encoders(embeddings, sparse_embeddings):
    """exactly what `cfg` declares, so `require_encoders` lets it through."""
    return Encoders(
        dense=embeddings, dense_model=FAKE_DENSE_MODEL,
        sparse=sparse_embeddings, sparse_model=FAKE_SPARSE_MODEL,
    )


@pytest.fixture
def dense_encoders(embeddings):
    """what `dense_cfg` declares. also the stand-in for 'forgot the sparse encoder'."""
    return Encoders(dense=embeddings, dense_model=FAKE_DENSE_MODEL)