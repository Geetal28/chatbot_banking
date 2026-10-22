"""Hybrid retrieval: dense vectors (Chroma) + BM25, fused with Reciprocal Rank Fusion,
with an optional cross-encoder reranker.

Why hybrid: dense embeddings are good at meaning ("cost of going overdrawn") but weak at
exact tokens ("Section 5.2", "CHAPS", a product name). BM25 is the opposite. Banking
documents are full of both, so fusing the two lists is a cheap, well-tested win.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Plus

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")
RRF_K = 60  # standard constant from the original RRF paper
# BM25Plus rather than BM25Okapi: Okapi's IDF goes to zero or negative for terms that appear in
# half or more of the chunks, which silently breaks small corpora (a single short PDF).


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Embedder(Protocol):
    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers. Imported lazily so tests don't need torch."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # bge English v1.5 models are trained with this instruction on the query side only.
        self._query_prefix = (
            "Represent this sentence for searching relevant passages: "
            if "bge" in model_name.lower() and "en" in model_name.lower()
            else ""
        )

    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if is_query and self._query_prefix:
            texts = [self._query_prefix + t for t in texts]
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, p) for p in passages])]


@dataclass
class Hit:
    id: str
    text: str
    source: str
    page: int
    doc_type: str
    dense_score: float | None = None  # cosine similarity, None if not retrieved densely
    bm25_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        index_dir: Path,
        collection: str = "bank_docs",
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.embedder = embedder
        self.reranker = reranker
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(index_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._name = collection
        self._col = self._client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        self._bm25: BM25Plus | None = None
        self._bm25_ids: list[str] = []
        self._store: dict[str, tuple[str, dict[str, Any]]] = {}

    # ---------- indexing ----------
    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._col = self._client.get_or_create_collection(self._name, metadata={"hnsw:space": "cosine"})
        self._bm25 = None

    def count(self) -> int:
        return self._col.count()

    def add_chunks(self, ids: list[str], texts: list[str], metadatas: list[dict[str, Any]], batch: int = 64) -> None:
        """Upsert by deterministic id, so re-running ingestion never creates duplicates."""
        for i in range(0, len(ids), batch):
            sl = slice(i, i + batch)
            vecs = self.embedder.embed(texts[sl], is_query=False)
            self._col.upsert(ids=ids[sl], embeddings=vecs, documents=texts[sl], metadatas=metadatas[sl])
        self._bm25 = None  # corpus changed, rebuild lazily

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        data = self._col.get(include=["documents", "metadatas"])
        self._bm25_ids = list(data["ids"])
        self._store = {i: (d, m) for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])}
        corpus = [tokenize(self._store[i][0]) for i in self._bm25_ids]
        self._bm25 = BM25Plus(corpus) if corpus else None

    # ---------- search ----------
    def _dense(self, query: str, k: int) -> list[tuple[str, float]]:
        if self.count() == 0:
            return []
        qv = self.embedder.embed([query], is_query=True)[0]
        res = self._col.query(query_embeddings=[qv], n_results=min(k, self.count()), include=["distances"])
        return [(i, 1.0 - float(d)) for i, d in zip(res["ids"][0], res["distances"][0])]

    def _sparse(self, query: str, k: int) -> list[tuple[str, float]]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda x: x[1], reverse=True)[:k]
        return [(i, float(s)) for i, s in ranked if s > 0]

    def search(self, query: str, top_k: int = 5, fetch_k: int = 20, mode: str = "hybrid", rerank: bool | None = None) -> list[Hit]:
        """mode: 'dense' | 'bm25' | 'hybrid'. Exposed so the eval script can compare them."""
        self._ensure_bm25()
        dense = self._dense(query, fetch_k) if mode in {"dense", "hybrid"} else []
        sparse = self._sparse(query, fetch_k) if mode in {"bm25", "hybrid"} else []
        dense_map, sparse_map = dict(dense), dict(sparse)

        fused: dict[str, float] = {}
        for ranking in (dense, sparse):
            for rank, (doc_id, _s) in enumerate(ranking, start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)

        order = sorted(fused, key=fused.get, reverse=True)
        hits: list[Hit] = []
        for doc_id in order[: max(fetch_k, top_k)]:
            text, meta = self._store.get(doc_id) or self._fetch(doc_id)
            hits.append(Hit(
                id=doc_id, text=text, source=str(meta.get("source", "?")), page=int(meta.get("page", 0)),
                doc_type=str(meta.get("doc_type", "general")), dense_score=dense_map.get(doc_id),
                bm25_score=sparse_map.get(doc_id), fused_score=fused[doc_id], meta=meta,
            ))

        use_rerank = (self.reranker is not None) if rerank is None else (rerank and self.reranker is not None)
        if use_rerank and hits:
            scores = self.reranker.score(query, [h.text for h in hits])  # type: ignore[union-attr]
            for h, s in zip(hits, scores):
                h.rerank_score = s
            hits.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
        return hits[:top_k]

    def _fetch(self, doc_id: str) -> tuple[str, dict[str, Any]]:
        r = self._col.get(ids=[doc_id], include=["documents", "metadatas"])
        return r["documents"][0], r["metadatas"][0]

    def best_dense_score(self, query: str) -> float:
        """Top cosine similarity for a query. Used for the 'not in the documents' decision.

        Computed separately from the fused ranking because RRF scores are rank-based
        and say nothing about absolute relevance.
        """
        d = self._dense(query, 1)
        return d[0][1] if d else 0.0
