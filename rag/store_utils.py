"""
store_utils.py

Convenience loaders for the persisted arXiv corpus + its embedding client,
used wherever the LangGraph pipeline needs to query the RAG store at
runtime (setup_arxiv_db.py builds/extends the corpus; this just loads it).
"""

from __future__ import annotations

import logging
from pathlib import Path

from openai import OpenAI

from rag.embed_store import ArxivVectorStore
from functools import lru_cache

logger = logging.getLogger("store_utils")

DEFAULT_INDEX_NAME = "qaoa-portfolio-v1"
DEFAULT_DB_DIR = "data/references/arxiv_db"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIM = 1536


def load_arxiv_store(
    index_name: str = DEFAULT_INDEX_NAME,
    db_dir: str | Path = DEFAULT_DB_DIR,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> ArxivVectorStore | None:
    """Load the persisted arXiv store. Returns None (not a raised error) if
    no corpus exists yet — matches the original notebook's behavior of
    treating a missing corpus as 'skip this node', not a hard failure."""
    store = ArxivVectorStore(index_name, Path(db_dir), embedding_dim)
    try:
        loaded = store.load_or_init()
    except Exception as e:
        logger.warning(f"Could not load arXiv corpus ({e}). This node will be skipped.")
        return None
    if not loaded:
        logger.warning(
            f"No arXiv corpus found at '{db_dir}/{index_name}.*' — "
            f"run setup_arxiv_db.py first. This node will be skipped."
        )
        return None
    logger.info(f"Reloaded: {len(store)} chunks, {len(store.manifest)} papers.")
    return store


def get_arxiv_runtime(
    index_name: str = DEFAULT_INDEX_NAME,
    db_dir: str | Path = DEFAULT_DB_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> tuple[ArxivVectorStore | None, OpenAI, str]:
    """Load the store + construct the OpenAI embedding client together,
    since every node that queries the corpus needs both. Requires
    OPENAI_API_KEY already in the environment (call load_dotenv() first)."""
    store = load_arxiv_store(index_name, db_dir, embedding_dim)
    client = OpenAI()
    return store, client, embedding_model

@lru_cache(maxsize=1)
def get_arxiv_runtime_cached():
    """Same as get_arxiv_runtime(), but loaded once and reused for the
    lifetime of the process — avoids re-reading the FAISS index from disk
    on every node call within a single graph run."""
    return get_arxiv_runtime()

# rag/store_utils.py 
from rag.doc_lookup import QiskitDocStore

DEFAULT_QISKIT_DOC_STORE_PATH = "data/references/qiskit_formatted.json"

@lru_cache(maxsize=1)
def get_qiskit_doc_store(path: str = DEFAULT_QISKIT_DOC_STORE_PATH) -> QiskitDocStore | None:
    try:
        store = QiskitDocStore(path)
        logger.info(f"Qiskit doc store loaded: {len(store)} classes.")
        return store
    except Exception as e:
        logger.warning(
            f"Could not load Qiskit doc store ({e}). Static validation will be "
            f"skipped — run setup_qiskit_db.py first to enable it. Node still works without it."
        )
        return None