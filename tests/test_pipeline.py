"""End-to-end tests using a fake Ollama client (no model needed)."""
import re
import zlib
from pathlib import Path

import pytest

from localrag.config import Config
from localrag.pipeline import NOT_FOUND, answer, cited_indices, ingest, retrieve
from localrag.store import VectorStore


class FakeClient:
    """Bag-of-words embeddings and a canned chat reply."""

    DIM = 512

    def embed(self, model, texts):
        out = []
        for t in texts:
            v = [0.0] * self.DIM
            for w in re.findall(r"[a-z]+", t.lower()):
                v[zlib.crc32(w.encode()) % self.DIM] += 1.0
            if not any(v):
                v[0] = 1.0
            out.append(v)
        return out

    def chat(self, model, messages, **kwargs):
        yield "The answer is in the context [1]."

    def chat_text(self, *a, **k):
        return "".join(self.chat(*a, **k))


@pytest.fixture
def env(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ferrets.txt").write_text("Ferrets sleep fourteen to eighteen hours every day and eat meat.")
    (docs / "sailing.md").write_text("Sailing boats use a keel and rudder to steer through water and wind.")
    cfg = Config(
        docs_dir=str(docs),
        db_dir=str(tmp_path / "db"),
        query_prefix="",
        doc_prefix="",
        min_similarity=0.2,
        condense_followups=False,
    )
    return cfg, FakeClient(), VectorStore(cfg.db_dir, "test")


def test_ingest_then_retrieve_finds_right_file(env):
    cfg, client, store = env
    stats = ingest(cfg, client, store, log=lambda _: None)
    assert stats.added == 2 and stats.failed == []
    hits = retrieve(cfg, client, store, "How long do ferrets sleep?")
    assert hits and hits[0].source == "ferrets.txt"


def test_second_ingest_skips_unchanged_and_updates_changed(env):
    cfg, client, store = env
    ingest(cfg, client, store, log=lambda _: None)
    again = ingest(cfg, client, store, log=lambda _: None)
    assert again.skipped == 2 and again.added == 0

    (Path(cfg.docs_dir) / "ferrets.txt").write_text("Ferrets love tunnels.")
    changed = ingest(cfg, client, store, log=lambda _: None)
    assert changed.updated == 1 and changed.skipped == 1


def test_deleted_files_are_pruned(env):
    cfg, client, store = env
    ingest(cfg, client, store, log=lambda _: None)
    (Path(cfg.docs_dir) / "sailing.md").unlink()
    stats = ingest(cfg, client, store, log=lambda _: None)
    assert stats.removed == 1
    assert "sailing.md" not in store.sources()


def test_answer_streams_with_sources(env):
    cfg, client, store = env
    ingest(cfg, client, store, log=lambda _: None)
    ans = answer(cfg, client, store, "How long do ferrets sleep?")
    assert not ans.refused
    assert cited_indices(ans.text()) == [1]


def test_refuses_when_nothing_similar(env):
    cfg, client, store = env
    cfg.min_similarity = 0.99
    ingest(cfg, client, store, log=lambda _: None)
    ans = answer(cfg, client, store, "quantum chromodynamics lagrangian")
    assert ans.refused and ans.text() == NOT_FOUND


def test_empty_store_refuses(env):
    cfg, client, store = env
    ans = answer(cfg, client, store, "anything")
    assert ans.refused


def test_reset_clears_index(env):
    cfg, client, store = env
    ingest(cfg, client, store, log=lambda _: None)
    store.reset()
    assert store.count() == 0


def test_eval_harness_runs(env, tmp_path):
    from localrag.evaluate import run_eval

    cfg, client, store = env
    ingest(cfg, client, store, log=lambda _: None)
    qs = tmp_path / "q.jsonl"
    qs.write_text(
        '{"question": "How long do ferrets sleep?", "expected_sources": ["ferrets.txt"], "expected_keywords": ["answer"]}\n'
    )
    report = run_eval(cfg, client, store, str(qs), log=lambda _: None)
    assert report["summary"]["retrieval_hit_rate"] == 1.0
    assert report["summary"]["answer_accuracy"] == 1.0
