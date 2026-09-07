import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient

from config import IndexConfig

DIMENSION = 32


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
    return IndexConfig(base="test_docs", dataset="fake/dataset", description="fixture corpus")


@pytest.fixture
def loader():
    def _load(cfg):
        return [
            Document(page_content=f"chunk {i} about transformers and pipelines",
                     metadata={"source": f"doc-{i}"})
            for i in range(6)
        ]
    return _load