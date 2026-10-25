from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .ollama import OllamaClient, OllamaError
from .pipeline import NOT_FOUND, answer, ingest
from .store import VectorStore


def _services(args, **overrides):
    cfg = load_config(args.config)
    if overrides:
        cfg = replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    client = OllamaClient(cfg.ollama_host)
    return cfg, client


def _check_models(cfg, client, need_chat=True):
    needed = [cfg.embed_model] + ([cfg.chat_model] if need_chat else [])
    missing = client.missing_models(*needed)
    if missing:
        cmds = "\n".join(f"  ollama pull {m}" for m in missing)
        raise OllamaError(f"Missing models. Install them with:\n{cmds}")


def _print_sources(hits, answered=True):
    if not hits:
        return
    print("\nSources:" if answered else "\nClosest passages found (none of them answered it):")
    for i, h in enumerate(hits, start=1):
        snippet = h.text.replace("\n", " ")[:110]
        print(f"  [{i}] {h.label}  (similarity {h.score:.2f})\n      {snippet}...")


def cmd_doctor(args):
    cfg, client = _services(args)
    print(f"Ollama host : {cfg.ollama_host}")
    models = client.list_models()
    print(f"Installed   : {', '.join(models) or '(none)'}")
    missing = client.missing_models(cfg.chat_model, cfg.embed_model)
    if missing:
        print("\nMissing models:")
        for m in missing:
            print(f"  ollama pull {m}")
        return 1
    print("\nAll good: Ollama is reachable and both models are installed.")
    return 0


def cmd_ingest(args):
    cfg, client = _services(args, docs_dir=args.docs)
    _check_models(cfg, client, need_chat=False)
    store = VectorStore(cfg.db_dir, cfg.collection)
    print(f"Indexing {cfg.docs_dir} ...")
    s = ingest(cfg, client, store, force=args.force)
    print(
        f"\nDone. added={s.added} updated={s.updated} skipped={s.skipped} "
        f"removed={s.removed} new_chunks={s.chunks} failed={len(s.failed)}"
    )
    print(f"Total chunks in database: {store.count()}")
    return 0


def cmd_ask(args):
    cfg, client = _services(args)
    _check_models(cfg, client)
    store = VectorStore(cfg.db_dir, cfg.collection)
    ans = answer(cfg, client, store, args.question, k=args.top_k)
    reply = []
    for token in ans.stream:
        reply.append(token)
        print(token, end="", flush=True)
    print()
    _print_sources(ans.hits, answered=NOT_FOUND.lower() not in "".join(reply).lower())
    return 0


def cmd_chat(args):
    cfg, client = _services(args)
    _check_models(cfg, client)
    store = VectorStore(cfg.db_dir, cfg.collection)
    if store.count() == 0:
        print("The database is empty. Run `localrag ingest` first.")
        return 1
    print(f"Chatting with {store.count()} chunks using {cfg.chat_model}. Type 'exit' to quit.\n")
    history: list[dict] = []
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q"}:
            break
        ans = answer(cfg, client, store, q, history=history)
        print("bot > ", end="", flush=True)
        reply = []
        for token in ans.stream:
            reply.append(token)
            print(token, end="", flush=True)
        print()
        _print_sources(ans.hits, answered=NOT_FOUND.lower() not in "".join(reply).lower())
        print()
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": "".join(reply)}]
    return 0


def cmd_stats(args):
    cfg, _ = _services(args)
    store = VectorStore(cfg.db_dir, cfg.collection)
    sources = store.sources()
    print(f"{len(sources)} documents, {store.count()} chunks\n")
    for src, n in sorted(sources.items()):
        print(f"  {n:>5}  {src}")
    return 0


def cmd_reset(args):
    cfg, _ = _services(args)
    if not args.yes:
        if input(f"Delete the whole index at {cfg.db_dir}? [y/N] ").strip().lower() != "y":
            print("Cancelled.")
            return 1
    VectorStore(cfg.db_dir, cfg.collection).reset()
    print("Index cleared. Your original documents were not touched.")
    return 0


def cmd_eval(args):
    from .evaluate import run_eval, save_report

    cfg, client = _services(args)
    _check_models(cfg, client, need_chat=not args.no_generate)
    store = VectorStore(cfg.db_dir, cfg.collection)
    print(f"Evaluating with {args.questions} ...\n")
    report = run_eval(
        cfg, client, store, args.questions, generate=not args.no_generate, top_k=args.top_k
    )
    print("\nSummary")
    for k, v in report["summary"].items():
        print(f"  {k:<20} {v}")
    path = save_report(report)
    print(f"\nFull report saved to {path}")
    return 0


def cmd_serve(args):
    app = Path(__file__).with_name("webapp.py")
    return subprocess.call(
        [
            sys.executable, "-m", "streamlit", "run", str(app),
            "--server.address", "127.0.0.1",
            "--browser.gatherUsageStats", "false",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="localrag", description="Private, local chat with your documents.")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check Ollama and models").set_defaults(fn=cmd_doctor)

    s = sub.add_parser("ingest", help="index documents")
    s.add_argument("--docs", default=None, help="override docs_dir")
    s.add_argument("--force", action="store_true", help="re-index unchanged files too")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("ask", help="ask one question")
    s.add_argument("question")
    s.add_argument("--top-k", type=int, default=None)
    s.set_defaults(fn=cmd_ask)

    sub.add_parser("chat", help="interactive chat in the terminal").set_defaults(fn=cmd_chat)
    sub.add_parser("stats", help="show what is indexed").set_defaults(fn=cmd_stats)

    s = sub.add_parser("reset", help="delete the index")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_reset)

    s = sub.add_parser("eval", help="run the evaluation set")
    s.add_argument("--questions", default="eval/questions.example.jsonl")
    s.add_argument("--no-generate", action="store_true", help="retrieval-only (fast)")
    s.add_argument("--top-k", type=int, default=None)
    s.set_defaults(fn=cmd_eval)

    sub.add_parser("serve", help="launch the web UI").set_defaults(fn=cmd_serve)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        code = args.fn(args)
    except OllamaError as e:
        print(f"\nError: {e}", file=sys.stderr)
        code = 2
    except KeyboardInterrupt:
        code = 130
    sys.exit(code or 0)
