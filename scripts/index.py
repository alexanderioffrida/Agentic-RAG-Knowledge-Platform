# alias swap, build, cleanup, rollback

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.sparse_embeddings import SparseEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    CreateAlias,
    CreateAliasOperation,
    Distance,
    Modifier,
    SparseVectorParams,
    VectorParams
)

from config import DENSE_VECTOR, SPARSE_VECTOR, IndexConfig, require_env

BUILD_STAMP = "%Y%m%dT%H%M%S%fZ"
_BUILD_RE = re.compile(r"__build_(\d{8}T\d{12}Z)_[0-9a-f]{8}$")

_client: QdrantClient | None = None

def get_client():
    '''returns a single module-level QdrantClient, created on first use and reused.'''
    global _client
    if _client is None:
        require_env("QDRANT_URL", "QDRANT_KEY")
        _client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY"))
    return _client

def split_documents(docs: Iterable[Document], cfg: IndexConfig) -> list[Document]:
    '''splits doucments into manageable chunks, preserving boundary contex through overlap.'''
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        disallowed_special=()
    )
    return splitter.split_documents(list(docs))

def build_sparse_embeddings(cfg: IndexConfig) -> SparseEmbeddings | None:
    """the BM25 encoder, or None for a dense-only config. imported lazily."""
    if not cfg.hybrid:
        return None
    from langchain_qdrant import FastEmbedSparse

    return FastEmbedSparse(model_name=cfg.sparse_model)

def require_sparse_encoder(
    cfg: IndexConfig, sparse_embeddings: SparseEmbeddings | None
) -> None:
    """a hybrid config with no encoder retrieves dense-only and reports nothing. refuse it."""
    if cfg.hybrid and sparse_embeddings is None:
        raise ValueError(
            f"'{cfg.alias}' is configured for hybrid retrieval "
            f"(sparse_model={cfg.sparse_model!r}) but no sparse encoder was supplied"
        )

def collection_config(cfg: IndexConfig, dimension: int) -> dict:
    """vector configuration for a build collection."""
    sparse: dict[str, SparseVectorParams] = {}
    if cfg.hybrid:
        sparse[SPARSE_VECTOR] = SparseVectorParams(modifier=Modifier.IDF)
    return {
        "vectors_config": {
            DENSE_VECTOR: VectorParams(size=dimension, distance=Distance.COSINE)
        },
        "sparse_vectors_config": sparse
    }

def load_hf_dataset(cfg: IndexConfig) -> list[Document]:
    """default doc source."""
    from langchain_community.document_loaders import HuggingFaceDatasetLoader

    loader = HuggingFaceDatasetLoader(cfg.dataset, cfg.content_column)
    return loader.load()[: cfg.n_docs]

# ––– Build Naming –––

def _build_name(cfg: IndexConfig) -> str:
    """`{alias}__build_{utc_timestamp}_{uuid8}`"""
    stamp = datetime.now(timezone.utc).strftime(BUILD_STAMP)
    return f"{cfg.build_prefix}{stamp}_{uuid.uuid4().hex[:8]}"

def _build_age(name: str) -> timedelta | None:
    match = _BUILD_RE.search(name)
    if match is None:
        return None
    created = datetime.strptime(match.group(1), BUILD_STAMP).replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - created

# ––– Alias Inspect –––

def current_target(client: QdrantClient, alias: str) -> str | None:
    """the collection an alias currently points at, or None if the alias is absent."""
    for entry in client.get_aliases().aliases:
        if entry.alias_name == alias:
            return entry.collection_name
    return None

def list_builds(client: QdrantClient, cfg: IndexConfig) -> list[str]:
    """build collections for this alias, newest first."""
    names = [
        collection.name
        for collection in client.get_collections().collections
        if collection.name.startswith(cfg.build_prefix)
    ]
    return sorted(names, reverse=True)

# ––– Lifecycle –––

def swap_alias(client: QdrantClient, alias: str, build: str) -> None:
    '''points the alias at the completed build via one CreateAliasOperation. this uses
    the exact same alias name and leverages the upsert behavior to modify where it points.'''
    client.update_collection_aliases(
        change_aliases_operations=[
            CreateAliasOperation(
                create_alias=CreateAlias(collection_name=build, alias_name=alias)
            )
        ]
    )

def cleanup_builds(
    client: QdrantClient, 
    cfg: IndexConfig,
    keep: int = 1,
    min_age_minutes: int = 30
    ) -> list[str]:
    '''deletes every {alias}__build_* collection that is not the alias's current target. returns the names deleted.'''
    current = current_target(client, cfg.alias)
    retain = keep if current else 0
    superseded = [name for name in list_builds(client, cfg) if name != current]

    deleted: list[str] = []
    for name in superseded[retain:]:
        age = _build_age(name)
        if age is not None and age < timedelta(minutes=min_age_minutes):
            continue
        print(f"-> Removing stale build collection '{name}'...")
        client.delete_collection(name)
        deleted.append(name)
    return deleted

def build_index(
    client: QdrantClient,
    cfg: IndexConfig,
    embeddings: Embeddings,
    sparse_embeddings: SparseEmbeddings | None = None,
    loader: Callable[[IndexConfig], list[Document]] | None = None
) -> str:
    """creates a build collection, uploads splits, verifies the count, returns its name."""
    print(f"-> Building index for '{cfg.alias}' from '{cfg.dataset}'...")
    dimension = len(embeddings.embed_query("dimension_probe"))

    documents = (loader or load_hf_dataset)(cfg)
    splits = split_documents(documents, cfg)

    # deliberately placed before create_collection
    require_sparse_encoder(cfg, sparse_embeddings)

    build = _build_name(cfg)
    client.create_collection(
        build, **collection_config(cfg, dimension)
    )

    try:
        store = QdrantVectorStore(
            client=client, 
            collection_name=build, 
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID if cfg.hybrid else RetrievalMode.DENSE,
            vector_name=DENSE_VECTOR,
            sparse_vector_name=SPARSE_VECTOR
        )
        store.add_documents(splits)
        uploaded = client.count(build, exact=True).count
        if uploaded != len(splits):
            raise RuntimeError(
                f"expected {len(splits)} points in '{build}', found {uploaded}."
            )
    except BaseException:
        print(f"-> Ingest failed. Discarding incomplete build '{build}'...")
        client.delete_collection(build)
        raise
    
    print(f"-> Uploaded {len(splits)} chunks to '{build}'.")
    return build

def ensure_index(
    client: QdrantClient,
    cfg: IndexConfig,
    embeddings: Embeddings,
    sparse_embeddings: SparseEmbeddings | None = None,
    loader: Callable[[IndexConfig], list[Document]] | None = None
) -> str:
    """returns the alias, guaranteed to resolve to a complete index."""
    # the warm-start return below skips build_index, so the guard has to run first:
    # a hybrid alias was built with sparse vectors that dense-only retrieval never reads.
    require_sparse_encoder(cfg, sparse_embeddings)
    cleanup_builds(client, cfg)

    if client.collection_exists(cfg.alias):
        print(f"-> Alias '{cfg.alias}' found. Loading existing index...")
        return cfg.alias
    
    build = build_index(client, cfg, embeddings, sparse_embeddings=sparse_embeddings, loader=loader)
    swap_alias(client, cfg.alias, build)
    cleanup_builds(client, cfg)
    print(f"-> Alias '{cfg.alias}' now points at '{build}'")
    return cfg.alias

def rollback(client: QdrantClient, cfg: IndexConfig) -> str | None:
    """points the alias at the most recent retained build that is not the current one."""
    current = current_target(client, cfg.alias)
    for name in list_builds(client, cfg):
        if name != current:
            swap_alias(client, cfg.alias, name)
            print(f"-> Rolled '{cfg.alias}' back to '{name}'.")
            return name
    return None