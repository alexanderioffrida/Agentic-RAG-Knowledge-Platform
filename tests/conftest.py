import sys
from pathlib import Path
import pytest
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

# Ensure scripts directory and repo root are on sys.path
root_dir = Path(__file__).resolve().parent.parent
scripts_dir = root_dir / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts import main


class FakeLoader:
    '''fake dataset loader returning predictable documents for testing.'''
    def __init__(self, path: str, page_content_column: str = "text"):
        self.path = path
        self.page_content_column = page_content_column

    def load(self):
        return [
            Document(
                page_content=f"document {i} content about transformers and huggingface",
                metadata={"source": f"doc_{i}"}
            )
            for i in range(8)
        ]


@pytest.fixture
def memory_client():
    '''returns a fresh in-memory QdrantClient for isolated test runs.'''
    return QdrantClient(":memory:")


@pytest.fixture
def fake_embeddings():
    '''returns deterministic 8-dimensional fake embeddings.'''
    return DeterministicFakeEmbedding(size=8)


@pytest.fixture
def patch_main_dependencies(monkeypatch, fake_embeddings):
    '''patches external dependencies in main to use in-memory fixtures.'''
    monkeypatch.setattr(main, "HuggingFaceDatasetLoader", FakeLoader)
    monkeypatch.setattr(main, "OpenAIEmbeddings", lambda model=None: fake_embeddings)
    monkeypatch.setattr(main, "embedding_model", "text-embedding-3-small")
    monkeypatch.setattr(main, "number_of_docs", 50)
