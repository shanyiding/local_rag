"""Understand references like 'eco320', 'slide 3', or a file name inside a question."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

_PAGE_RE = re.compile(r"\b(?:slides?|pages?|pg|p)\.?\s*#?\s*(\d{1,4})\b", re.IGNORECASE)
_PAGE_WORDS = {"slide", "slides", "page", "pages", "pg", "p", "q", "question", "week", "lecture"}


@dataclass
class Scope:
    sources: list[str] | None = None  # restrict search to these files
    page: int | None = None           # a specific page / slide number


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _identifier_candidates(question: str) -> set[str]:
    """Tokens that look like course/file codes: letters AND digits, e.g. eco320, sta302, q3report."""
    tokens = re.findall(r"[A-Za-z]+|\d+", question)
    words = re.findall(r"[A-Za-z0-9]+", question)
    cands = set(words)
    # join "eco 320" -> "eco320" (but not "slide 3" / "page 12")
    for a, b in zip(tokens, tokens[1:]):
        if a.isalpha() and b.isdigit() and a.lower() not in _PAGE_WORDS and len(a) >= 2:
            cands.add(a + b)
    out = set()
    for c in cands:
        n = _norm(c)
        if len(n) >= 4 and any(ch.isdigit() for ch in n) and any(ch.isalpha() for ch in n):
            out.add(n)
    return out


def parse_scope(question: str, sources: list[str]) -> Scope:
    """Work out which files / page the question is pointing at (if any)."""
    page_match = _PAGE_RE.search(question)
    page = int(page_match.group(1)) if page_match else None

    norm_paths = {s: _norm(s) for s in sources}
    qn = _norm(question)

    # 1) the question names a file (its full name without extension)
    by_name = [
        s for s in sources
        if len(_norm(PurePosixPath(s).stem)) >= 8 and _norm(PurePosixPath(s).stem) in qn
    ]
    if by_name:
        return Scope(sources=by_name, page=page)

    # 2) the question contains identifiers like eco320 that appear in file paths
    groups = []
    for cand in _identifier_candidates(question):
        matched = {s for s, n in norm_paths.items() if cand in n}
        if matched:
            groups.append(matched)
    if groups:
        inter = set.intersection(*groups)
        chosen = inter if inter else set.union(*groups)
        return Scope(sources=sorted(chosen), page=page)

    return Scope(sources=None, page=page)
