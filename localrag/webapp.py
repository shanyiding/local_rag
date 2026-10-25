"""Streamlit web UI. Launch with: localrag serve"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import streamlit as st

from localrag.config import load_config
from localrag.ollama import OllamaClient, OllamaError
from localrag.pipeline import NOT_FOUND, answer, cited_indices, ingest
from localrag.store import VectorStore

st.set_page_config(page_title="LocalRAG", page_icon="📚", layout="wide")


@st.cache_resource
def get_services():
    cfg = load_config()
    return cfg, OllamaClient(cfg.ollama_host), VectorStore(cfg.db_dir, cfg.collection)


base_cfg, client, store = get_services()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📚 LocalRAG")
    st.caption("Everything runs on this machine.")

    try:
        installed = client.list_models()
    except OllamaError as e:
        st.error(str(e))
        st.stop()

    chat_options = [m for m in installed if "embed" not in m] or [base_cfg.chat_model]
    default_idx = next(
        (i for i, m in enumerate(chat_options) if m.split(":")[0] == base_cfg.chat_model.split(":")[0]), 0
    )
    chat_model = st.selectbox("Chat model", chat_options, index=default_idx)
    top_k = st.slider("Passages to retrieve (top-k)", 1, 12, base_cfg.top_k)
    min_sim = st.slider("Minimum similarity", 0.0, 0.9, base_cfg.min_similarity, 0.05)
    cfg = replace(base_cfg, chat_model=chat_model, top_k=top_k, min_similarity=min_sim)

    st.divider()
    st.subheader("Documents")
    uploads = st.file_uploader(
        "Add files", accept_multiple_files=True, type=[e.lstrip(".") for e in cfg.extensions]
    )
    if uploads:
        target = Path(cfg.docs_dir) / "uploads"
        target.mkdir(parents=True, exist_ok=True)
        for f in uploads:
            (target / f.name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(uploads)} file(s). Click Re-index below.")

    if st.button("Re-index documents", use_container_width=True):
        missing = client.missing_models(cfg.embed_model)
        if missing:
            st.error(f"Missing embedding model. Run: ollama pull {missing[0]}")
        else:
            lines: list[str] = []
            with st.spinner("Indexing..."):
                stats = ingest(cfg, client, store, log=lines.append)
            st.success(
                f"added {stats.added}, updated {stats.updated}, "
                f"skipped {stats.skipped}, removed {stats.removed}"
            )
            if lines:
                st.code("\n".join(lines))

    sources = store.sources()
    st.caption(f"{len(sources)} documents · {store.count()} chunks indexed")
    with st.expander("Indexed files"):
        for src, n in sorted(sources.items()):
            st.write(f"`{src}` ({n})")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------- chat
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(hits: list[dict], used: list[int], refused: bool = False):
    if not hits:
        return
    title = "Closest passages (none answered it)" if refused else f"Sources ({len(hits)})"
    with st.expander(title):
        for i, h in enumerate(hits, start=1):
            mark = " ✅ cited" if i in used else ""
            st.markdown(f"**[{i}] {h['label']}** · similarity {h['score']:.2f}{mark}")
            st.caption(h["text"][:600] + ("..." if len(h["text"]) > 600 else ""))


st.header("Ask your documents")
if store.count() == 0:
    st.info("Nothing indexed yet. Put files in the docs folder (or upload them) and click **Re-index documents**.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("hits", []), m.get("used", []), m.get("refused", False))

if prompt := st.chat_input("Ask a question about your documents..."):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            missing = client.missing_models(cfg.chat_model, cfg.embed_model)
            if missing:
                st.error(f"Missing model(s). Run: ollama pull {' && ollama pull '.join(missing)}")
                st.stop()
            ans = answer(cfg, client, store, prompt, history=history)
            reply = st.write_stream(ans.stream)
        except OllamaError as e:
            st.error(str(e))
            st.stop()
        used = cited_indices(reply)
        hits = [{"label": h.label, "score": h.score, "text": h.text} for h in ans.hits]
        refused = NOT_FOUND.lower() in reply.lower()
        render_sources(hits, used, refused)
    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "hits": hits, "used": used, "refused": refused}
    )
