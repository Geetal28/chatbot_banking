"""Central configuration. Everything comes from environment variables (or .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    groq_model: str
    embedding_model: str
    reranker_model: str
    use_reranker: bool
    docs_dir: Path
    index_dir: Path
    db_path: Path
    audit_log: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    fetch_k: int
    min_score: float


def get_settings() -> Settings:
    """Built on every call so tests can override env vars."""
    data = Path(os.getenv("DATA_DIR", ROOT / "data"))
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        use_reranker=_bool("USE_RERANKER", False),
        docs_dir=Path(os.getenv("DOCS_DIR", data / "docs")),
        index_dir=Path(os.getenv("INDEX_DIR", data / "index")),
        db_path=Path(os.getenv("DB_PATH", data / "bank_demo.sqlite")),
        audit_log=Path(os.getenv("AUDIT_LOG", data / "audit" / "events.jsonl")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        top_k=int(os.getenv("TOP_K", "5")),
        fetch_k=int(os.getenv("FETCH_K", "20")),
        min_score=float(os.getenv("MIN_SCORE", "0.35")),
    )
