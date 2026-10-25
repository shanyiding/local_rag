"""Small dependency-free BM25 keyword index (exact-term matching complements embeddings)."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_STOP = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was", "were",
    "what", "which", "who", "how", "does", "do", "did", "it", "this", "that", "with", "as",
    "at", "by", "be", "from", "me", "my", "i", "about", "tell", "give", "show",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lens = [len(d) for d in docs]
        self.avg = (sum(self.lens) / self.n) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))

    def scores(self, query_tokens: list[str]) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg or 1))
                out[i] += idf * tf * (self.k1 + 1) / norm
        return out
