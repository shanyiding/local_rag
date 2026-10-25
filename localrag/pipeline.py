"""Ingestion and question answering: the heart of the RAG system."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from .chunking import chunk_text
from .config import Config
from .loaders import LEGACY_FORMATS, load_file
from .scope import parse_scope
from .store import Hit, VectorStore

# Bump when the way chunks are embedded changes; old indexes then re-index automatically.
INDEX_FORMAT = 2
MAX_DIRECT_FILES = 3

NOT_FOUND = "I couldn't find that in your documents."

SYSTEM_PROMPT = f"""You are a careful assistant that answers questions using ONLY the numbered context excerpts provided by the user.

Rules:
1. Base every statement on the excerpts. Do not use outside knowledge.
2. Cite the excerpts you used with their numbers in square brackets, like [1] or [2][3].
3. If the excerpts do not contain the answer, reply exactly: {NOT_FOUND}
4. If the excerpts only partly answer the question, answer the part you can and say what is missing.
5. Be concise and direct."""


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------
@dataclass
class IngestStats:
    added: int = 0
    updated: int = 0
    skipped: int = 0
    removed: int = 0
    chunks: int = 0
    failed: list = field(default_factory=list)


def discover_files(cfg: Config) -> list[Path]:
    root = Path(cfg.docs_dir)
    if not root.exists():
        return []
    exts = {e.lower() for e in cfg.extensions}
    files = []
    for p in sorted(root.rglob("*")):
        rel_parts = p.relative_to(root).parts
        if any(part.startswith(".") for part in rel_parts) or p.name.startswith("~$"):
            continue
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def legacy_files(cfg: Config) -> list[Path]:
    """Old binary Office files (.doc/.ppt/.xls) that we can't read."""
    root = Path(cfg.docs_dir)
    if not root.exists():
        return []
    return [
        p for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in LEGACY_FORMATS and not p.name.startswith("~$")
    ]


def _header(source: str, locator: str) -> str:
    return source + (f" ({locator})" if locator else "")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ingest(
    cfg: Config,
    client,
    store: VectorStore,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> IngestStats:
    """Index every supported file in docs_dir. Unchanged files are skipped."""
    stats = IngestStats()
    root = Path(cfg.docs_dir)
    files = discover_files(cfg)
    on_disk = set()

    for old in legacy_files(cfg):
        new_ext = LEGACY_FORMATS[old.suffix.lower()]
        log(f"  ! {old.relative_to(root).as_posix()}: old format not supported, re-save as {new_ext}")
        stats.failed.append((old.relative_to(root).as_posix(), "legacy format"))

    for path in files:
        source = path.relative_to(root).as_posix()
        on_disk.add(source)
        try:
            digest = _sha256(path)
            existing, existing_fmt = store.file_info(source)
            if existing == digest and existing_fmt == INDEX_FORMAT and not force:
                stats.skipped += 1
                continue

            pieces: list[tuple[str, int, str]] = []  # (chunk text, page or -1, locator)
            for section in load_file(path):
                for chunk in chunk_text(section.text, cfg.chunk_size, cfg.chunk_overlap):
                    pieces.append((chunk, section.page or -1, section.locator or ""))
            if not pieces:
                log(f"  ! {source}: no extractable text, skipping")
                stats.failed.append((source, "no extractable text"))
                continue

            if existing is not None:
                store.delete_source(source)

            ids, docs, embs, metas = [], [], [], []
            for start in range(0, len(pieces), cfg.embed_batch_size):
                batch = pieces[start : start + cfg.embed_batch_size]
                # Prepend the file path (and page/slide) so the embedding "knows" which
                # document a chunk comes from. The stored text stays clean.
                vectors = client.embed(
                    cfg.embed_model,
                    [cfg.doc_prefix + _header(source, loc) + "\n" + text for text, _, loc in batch],
                )
                for offset, ((text, page, locator), vec) in enumerate(zip(batch, vectors)):
                    i = start + offset
                    ids.append(f"{digest[:16]}:{source}:{i}")
                    docs.append(text)
                    embs.append(vec)
                    metas.append(
                        {
                            "source": source, "page": page, "locator": locator,
                            "chunk_index": i, "file_hash": digest, "fmt": INDEX_FORMAT,
                        }
                    )
            store.add(ids, docs, embs, metas)

            stats.chunks += len(pieces)
            if existing is None:
                stats.added += 1
                log(f"  + {source}: {len(pieces)} chunks")
            else:
                stats.updated += 1
                log(f"  ~ {source}: re-indexed, {len(pieces)} chunks")
        except Exception as e:  # keep going; report at the end
            stats.failed.append((source, str(e)))
            log(f"  ! {source}: {e}")

    # Remove files that were deleted from disk.
    for source in set(store.sources()) - on_disk:
        store.delete_source(source)
        stats.removed += 1
        log(f"  - {source}: removed (file no longer exists)")

    return stats


# --------------------------------------------------------------------------
# Retrieval + answering
# --------------------------------------------------------------------------
def _fuse(vec_hits: list[Hit], kw_hits: list[Hit], k: int) -> list[Hit]:
    """Reciprocal-rank fusion: chunks ranked well by meaning OR by exact keywords rise."""
    score: dict[str, float] = {}
    by_id: dict[str, Hit] = {}
    for hits in (vec_hits, kw_hits):
        for rank, h in enumerate(hits):
            score[h.id] = score.get(h.id, 0.0) + 1.0 / (60 + rank)
            by_id.setdefault(h.id, h)
    order = sorted(score, key=score.get, reverse=True)[:k]
    return [by_id[i] for i in order]


def retrieve(
    cfg: Config,
    client,
    store: VectorStore,
    question: str,
    k: int | None = None,
    apply_threshold: bool = True,
) -> list[Hit]:
    k = k or cfg.top_k
    scope = parse_scope(question, list(store.sources()))
    vec = client.embed(cfg.embed_model, [cfg.query_prefix + question])[0]

    pool = max(k, cfg.candidate_pool) if cfg.hybrid else k
    vec_hits = store.query(vec, pool, sources=scope.sources)
    if cfg.hybrid:
        kw_hits = store.keyword_search(question, pool, sources=scope.sources)
        hits = _fuse(vec_hits, kw_hits, k)
        # Keyword-only hits have no vector score yet; compute it so the threshold is consistent.
        known = {h.id for h in vec_hits}
        missing = [h.id for h in hits if h.id not in known]
        if missing:
            sims = store.similarities(vec, missing)
            for h in hits:
                if h.id in sims:
                    h.score = sims[h.id]
    else:
        hits = vec_hits[:k]

    if apply_threshold:
        hits = [h for h in hits if h.score >= cfg.min_similarity]
    return hits


def condense_question(cfg: Config, client, question: str, history: list[dict] | None) -> str:
    """Turn a follow-up like 'what about the second one?' into a standalone query."""
    if not history or not cfg.condense_followups:
        return question
    transcript = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:])
    prompt = (
        "Given the conversation below and a follow-up question, rewrite the follow-up "
        "as a single standalone question that makes sense without the conversation. "
        "Output only the rewritten question.\n\n"
        f"Conversation:\n{transcript}\n\nFollow-up: {question}\n\nStandalone question:"
    )
    out = client.chat_text(
        cfg.chat_model,
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        num_ctx=cfg.num_ctx,
    ).strip()
    return out or question


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] (source: {h.label})\n{h.text}" for i, h in enumerate(hits, start=1))


def build_messages(question: str, hits: list[Hit], history: list[dict] | None = None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])
    messages.append(
        {
            "role": "user",
            "content": f"Context excerpts:\n\n{format_context(hits)}\n\nQuestion: {question}",
        }
    )
    return messages


@dataclass
class Answer:
    hits: list[Hit]
    stream: Iterator[str]
    standalone_question: str

    @property
    def refused(self) -> bool:
        return not self.hits

    def text(self) -> str:
        """Consume the stream and return the full answer (can only be called once)."""
        return "".join(self.stream)


def answer(
    cfg: Config,
    client,
    store: VectorStore,
    question: str,
    history: list[dict] | None = None,
    k: int | None = None,
) -> Answer:
    standalone = condense_question(cfg, client, question, history)

    # "summarize slide 3 of eco320": fetch that exact page instead of guessing by similarity.
    scope = parse_scope(standalone, list(store.sources()))
    if scope.page and scope.sources:
        direct = store.get_page_chunks(scope.sources, scope.page)
        files = sorted({h.source for h in direct})
        if not direct:
            msg = f"The matching files have no page or slide {scope.page}."
            return Answer(hits=[], stream=iter([msg]), standalone_question=standalone)
        if len(files) > MAX_DIRECT_FILES:
            shown = "\n".join(f"  - {f}" for f in files[:15])
            more = f"\n  ...and {len(files) - 15} more" if len(files) > 15 else ""
            msg = (
                f"That matches {len(files)} files that each have a page or slide {scope.page}:\n"
                f"{shown}{more}\n\nTell me which one (include part of its file name) and I'll use it."
            )
            return Answer(hits=[], stream=iter([msg]), standalone_question=standalone)
        hits = direct[:10]
        stream = client.chat(
            cfg.chat_model,
            build_messages(question, hits, history),
            temperature=cfg.temperature,
            num_ctx=cfg.num_ctx,
        )
        return Answer(hits=hits, stream=stream, standalone_question=standalone)

    hits = retrieve(cfg, client, store, standalone, k=k)
    if not hits:
        return Answer(hits=[], stream=iter([NOT_FOUND]), standalone_question=standalone)
    stream = client.chat(
        cfg.chat_model,
        build_messages(question, hits, history),
        temperature=cfg.temperature,
        num_ctx=cfg.num_ctx,
    )
    return Answer(hits=hits, stream=stream, standalone_question=standalone)


def cited_indices(text: str) -> list[int]:
    """Citation numbers like [1] or [2][3] that appear in an answer."""
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
