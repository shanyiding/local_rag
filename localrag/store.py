from __future__ import annotations

import math
from dataclasses import dataclass

import chromadb

from .bm25 import BM25, tokenize


@dataclass
class Hit:
    text: str
    source: str
    page: int | None
    chunk_index: int
    score: float  # cosine similarity, higher is better
    locator: str | None = None  # e.g. "p.3", "slide 4", "sheet Budget"
    id: str = ""

    @property
    def label(self) -> str:
        loc = self.locator or (f"p.{self.page}" if self.page else "")
        return f"{self.source}" + (f" ({loc})" if loc else "")


def _hit(chunk_id: str, doc: str, meta: dict, score: float) -> Hit:
    page = meta.get("page", -1)
    return Hit(
        id=chunk_id,
        text=doc,
        source=meta["source"],
        page=page if page is not None and page >= 0 else None,
        chunk_index=meta.get("chunk_index", 0),
        score=score,
        locator=meta.get("locator") or None,
    )


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class VectorStore:
    """ChromaDB wrapper: vector search, keyword (BM25) search and metadata filtering."""

    def __init__(self, path: str, collection: str = "docs"):
        self._name = collection
        self.client = chromadb.PersistentClient(path=path)
        self.col = self._open()
        self._invalidate()

    def _open(self):
        return self.client.get_or_create_collection(
            self._name, metadata={"hnsw:space": "cosine"}
        )

    def _invalidate(self) -> None:
        self._bm25 = None
        self._chunks: list[tuple[str, str, dict]] = []
        self._sources: dict[str, int] | None = None

    # -- bookkeeping -------------------------------------------------------
    def count(self) -> int:
        return self.col.count()

    def file_info(self, source: str) -> tuple[str | None, int | None]:
        """(file_hash, index format version) for an indexed source, or (None, None)."""
        res = self.col.get(where={"source": source}, limit=1, include=["metadatas"])
        metas = res["metadatas"]
        if not metas:
            return None, None
        return metas[0].get("file_hash"), metas[0].get("fmt")

    def file_hash(self, source: str) -> str | None:
        return self.file_info(source)[0]

    def sources(self) -> dict[str, int]:
        """Map of source path -> number of chunks."""
        if self._sources is None:
            res = self.col.get(include=["metadatas"])
            counts: dict[str, int] = {}
            for m in res["metadatas"]:
                counts[m["source"]] = counts.get(m["source"], 0) + 1
            self._sources = counts
        return self._sources

    def delete_source(self, source: str) -> None:
        self.col.delete(where={"source": source})
        self._invalidate()

    def reset(self) -> None:
        self.client.delete_collection(self._name)
        self.col = self._open()
        self._invalidate()

    # -- write -------------------------------------------------------------
    def add(self, ids, documents, embeddings, metadatas) -> None:
        self.col.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        self._invalidate()

    # -- read --------------------------------------------------------------
    @staticmethod
    def build_where(sources=None, page=None):
        clauses = []
        if sources:
            clauses.append({"source": {"$in": list(sources)}})
        if page is not None:
            clauses.append({"page": {"$eq": page}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def query(self, embedding: list[float], k: int, sources=None) -> list[Hit]:
        total = self.count()
        if total == 0:
            return []
        kwargs = dict(
            query_embeddings=[embedding],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )
        where = self.build_where(sources)
        if where:
            kwargs["where"] = where
        res = self.col.query(**kwargs)
        return [
            _hit(i, doc, meta, 1.0 - dist)
            for i, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def get_page_chunks(self, sources: list[str], page: int) -> list[Hit]:
        """Every chunk on one page/slide of the given files, in reading order."""
        res = self.col.get(
            where=self.build_where(sources, page), include=["documents", "metadatas"]
        )
        hits = [_hit(i, d, m, 1.0) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
        return sorted(hits, key=lambda h: (h.source, h.chunk_index))

    def keyword_search(self, query: str, k: int, sources=None) -> list[Hit]:
        """BM25 over chunk text plus its file path. Scores are rescored by the caller."""
        if self._bm25 is None:
            res = self.col.get(include=["documents", "metadatas"])
            self._chunks = list(zip(res["ids"], res["documents"], res["metadatas"]))
            self._bm25 = BM25([tokenize(m["source"] + " " + d) for _, d, m in self._chunks])
        allowed = set(sources) if sources else None
        scored = [
            (s, i)
            for i, s in self._bm25.scores(tokenize(query)).items()
            if allowed is None or self._chunks[i][2]["source"] in allowed
        ]
        scored.sort(reverse=True)
        return [_hit(*self._chunks[i], 0.0) for _, i in scored[:k]]

    def similarities(self, embedding: list[float], ids: list[str]) -> dict[str, float]:
        """Cosine similarity between a query vector and specific stored chunks."""
        if not ids:
            return {}
        res = self.col.get(ids=ids, include=["embeddings"])
        return {i: _cosine(embedding, e) for i, e in zip(res["ids"], res["embeddings"])}
