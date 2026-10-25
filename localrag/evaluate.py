"""Tiny evaluation harness so you measure your system instead of trusting it.

Each line of the questions file is a JSON object:
  {"question": "...",
   "expected_sources": ["notes/a.md"],     # optional: files that hold the answer
   "expected_keywords": ["800", "150"],    # optional: all must appear in the answer
   "should_refuse": false}                 # optional: true if docs don't contain it
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import Config
from .pipeline import NOT_FOUND, answer, retrieve
from .store import VectorStore


def load_questions(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def _rate(num: int, den: int):
    return round(num / den, 3) if den else None


def run_eval(
    cfg: Config,
    client,
    store: VectorStore,
    questions_path: str,
    generate: bool = True,
    top_k: int | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    questions = load_questions(questions_path)
    rows = []
    counts = {k: [0, 0] for k in ("retrieval", "answer", "refusal", "false_refusal")}

    for q in questions:
        text = q["question"]
        should_refuse = q.get("should_refuse", False)
        t0 = time.time()

        raw_hits = retrieve(cfg, client, store, text, k=top_k, apply_threshold=False)
        kept = [h for h in raw_hits if h.score >= cfg.min_similarity]
        row = {"question": text, "top_score": round(raw_hits[0].score, 3) if raw_hits else None}

        # Retrieval: did the right file show up in the top-k?
        expected_sources = q.get("expected_sources")
        if expected_sources and not should_refuse:
            ok = any(h.source in expected_sources for h in raw_hits)
            counts["retrieval"][0] += ok
            counts["retrieval"][1] += 1
            row["retrieval_ok"] = ok

        # Generation: keywords present, refusals correct.
        if generate:
            ans = answer(cfg, client, store, text, k=top_k)
            reply = ans.text()
            refused = NOT_FOUND.lower() in reply.lower()
            row["answer"] = reply
            if should_refuse:
                counts["refusal"][0] += refused
                counts["refusal"][1] += 1
                row["refusal_ok"] = refused
            else:
                counts["false_refusal"][0] += refused
                counts["false_refusal"][1] += 1
                keywords = q.get("expected_keywords")
                if keywords:
                    ok = (not refused) and all(k.lower() in reply.lower() for k in keywords)
                    counts["answer"][0] += ok
                    counts["answer"][1] += 1
                    row["answer_ok"] = ok
        else:
            refused = not kept
            if should_refuse:
                counts["refusal"][0] += refused
                counts["refusal"][1] += 1
                row["refusal_ok"] = refused
            else:
                counts["false_refusal"][0] += refused
                counts["false_refusal"][1] += 1

        row["seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        flags = [f"{k}={'PASS' if v else 'FAIL'}" for k, v in row.items() if k.endswith("_ok")]
        log(f"  {' '.join(flags) or 'n/a':<40} {text[:70]}")

    summary = {
        "retrieval_hit_rate": _rate(*counts["retrieval"]),
        "answer_accuracy": _rate(*counts["answer"]) if generate else None,
        "refusal_accuracy": _rate(*counts["refusal"]),
        "false_refusal_rate": _rate(*counts["false_refusal"]),
        "avg_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 2) if rows else None,
        "questions": len(rows),
    }
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {k: getattr(cfg, k) for k in (
            "chat_model", "embed_model", "chunk_size", "chunk_overlap", "top_k", "min_similarity"
        )},
        "summary": summary,
        "rows": rows,
    }
    return report


def save_report(report: dict, out_dir: str = "eval/results") -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stamp = report["timestamp"].replace(":", "-")
    path = Path(out_dir) / f"{stamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
