from __future__ import annotations

import re

_SEPARATORS = ("\n\n", "\n", ". ", "; ", " ")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks, preferring natural boundaries.

    Each chunk is at most `size` characters. We try to end a chunk at a paragraph
    break, then a line break, then a sentence end, then a space, so passages stay
    readable. Neighbouring chunks share `overlap` characters so an idea cut at a
    boundary still appears whole in at least one chunk.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = normalize(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            for sep in _SEPARATORS:
                idx = window.rfind(sep)
                if idx > size * 0.5:
                    end = start + idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks
