# knobs + index identity

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from numpy.dtypes import StrDType

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o"

RETRIEVAL_K = 5
CANDIDATE_LIMIT = 50
RRF_K = 60

def slug(value: str) -> str:
    """lowercases and collapses each run of non-alphanumerics into one underscore."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") # value is the embedding model

def require_env(*names: str) -> None:
    """fails loudly at startup rather than deep inside a client call."""
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            f"Missing required encironment variable(s): {', '.join(missing)}. "
            "Set them in .env or the shell environment."
        )

@dataclass(frozen=True)
class IndexConfig:
    """one corpus and every param that determines the index built from it."""
    base: str
    dataset: str
    description: str
    content_column: str = "text"
    n_docs: int = 50
    chunk_size: int = 700
    chunk_overlap: int = 50
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    @property
    def alias(self) -> str:
        return f"{self.base}__{slug(self.embedding_model)}__n{self.n_docs}"

    @property
    def build_prefix(self) -> str:
        return f"{self.alias}__build_"

CORPORA: tuple[IndexConfig, ...] = (
    IndexConfig(
        base="hf_docs",
        dataset="m-ric/huggingface_doc",
        description="HuggingFace documentation, including guides and Python code."
    ),
    IndexConfig(
        base="transformers_docs",
        dataset="m-ric/transformers_documentation_en",
        description="Documentation for the transformers library."
    )
)

    