from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    # Models
    ollama_host: str = "http://localhost:11434"
    chat_model: str = "llama3.2"
    embed_model: str = "nomic-embed-text"
    query_prefix: str = "search_query: "
    doc_prefix: str = "search_document: "
    # Storage
    docs_dir: str = "data/docs"
    db_dir: str = "data/chroma"
    collection: str = "docs"
    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150
    # Retrieval
    top_k: int = 8
    hybrid: bool = True
    candidate_pool: int = 30
    min_similarity: float = 0.30
    condense_followups: bool = True
    # Generation
    temperature: float = 0.2
    num_ctx: int = 4096
    # Ingestion
    embed_batch_size: int = 32
    extensions: list = field(
        default_factory=lambda: [
            ".txt", ".md", ".rst", ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm",
            ".py", ".csv", ".json",
        ]
    )


def load_config(path: str | None = None) -> Config:
    """Load config from YAML (if it exists), falling back to defaults."""
    path = path or os.environ.get("LOCALRAG_CONFIG", "config.yaml")
    data: dict = {}
    p = Path(path)
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    valid = {f.name for f in fields(Config)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    cfg = Config(**data)
    if cfg.chunk_overlap >= cfg.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    return cfg
