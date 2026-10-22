"""Streamlit front end.  Run:  streamlit run app/streamlit_app.py"""
from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run app/streamlit_app.py` puts app/ (not the project root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app import db as bankdb  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.graph import Assistant, get_llm  # noqa: E402
from app.ingest import SUPPORTED, build_index  # noqa: E402
from app.retrieval import CrossEncoderReranker, HybridRetriever, SentenceTransformerEmbedder  # noqa: E402

st.set_page_config(page_title="Retail Banking Assistant (demo)", page_icon="🏦", layout="wide")


@st.cache_resource(show_spinner="Loading models and index...")
def get_services():
    s = get_settings()
    bankdb.init_db(s.db_path)
    embedder = SentenceTransformerEmbedder(s.embedding_model)
    reranker = CrossEncoderReranker(s.reranker_model) if s.use_reranker else None
    retriever = HybridRetriever(embedder, s.index_dir, reranker=reranker)
    if retriever.count() == 0 and any(p.suffix.lower() in SUPPORTED and p.name.lower() != "readme.md" for p in s.docs_dir.glob("**/*")):
        build_index(s, retriever)
    llm = get_llm(s)
    return s, retriever, Assistant(retriever, llm, s)


settings, retriever, assistant = get_services()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Demo controls")
    customers = bankdb.list_customers(settings.db_path)
    labels = {c["customer_id"]: f'{c["name"]} ({c["customer_id"]})' for c in customers}
    customer_id = st.selectbox("Demo customer (synthetic data)", list(labels), format_func=labels.get)
    show_debug = st.toggle("Show routing, SQL and retrieval details", value=False)

    st.divider()
    st.subheader("Document index")
    st.caption(f"{retriever.count()} chunks indexed")
    uploads = st.file_uploader("Add public PDFs / text files", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploads and st.button("Save uploads"):
        settings.docs_dir.mkdir(parents=True, exist_ok=True)
        for f in uploads:
            (settings.docs_dir / f.name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(uploads)} file(s). Now rebuild the index.")
    if st.button("Rebuild index", type="primary"):
        with st.spinner("Indexing..."):
            stats = build_index(settings, retriever, reset=True)
        st.write(stats)

    st.divider()
    st.caption(
        f"Embeddings: `{settings.embedding_model}`  \n"
        f"LLM: `{settings.groq_model if assistant.llm_enabled else 'not configured'}`  \n"
        f"Reranker: `{'on' if settings.use_reranker else 'off'}`"
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- main ----------------
st.title("🏦 Retail Banking Assistant")
st.caption(
    "Demo project. Not affiliated with, or endorsed by, any bank. Answers come from public documents "
    "you load, and account data is synthetic. Not financial advice."
)
if not assistant.llm_enabled:
    st.warning("No GROQ_API_KEY set: running in retrieval-only mode (no generated answers).")
if retriever.count() == 0:
    st.info("The document index is empty. Add public documents to `data/docs/` (or upload them in the sidebar) and rebuild the index.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_extras(msg: dict) -> None:
    cites = [c for c in msg.get("citations", []) if c.get("cited")]
    if cites:
        with st.expander(f"Sources ({len(cites)})"):
            for c in cites:
                st.markdown(f"**[{c['n']}] {c['source']}**, page {c['page']} ({c['doc_type']})")
                st.caption(c["excerpt"])
    if show_debug:
        with st.expander("Debug"):
            st.write({"route": msg.get("route"), "latency_ms": msg.get("latency_ms"), "trace": msg.get("trace")})
            if msg.get("sql"):
                st.code(msg["sql"], language="sql")
            for c in msg.get("citations", []):
                st.caption(f"[{c['n']}] {c['source']} p.{c['page']}  dense={c['dense']}  bm25={c['bm25']}  rerank={c['rerank']}")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_extras(m)

if prompt := st.chat_input("Ask about fees, products, or your demo account..."):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = assistant.ask(prompt, customer_id, history)
        if result.get("pii_found"):
            st.caption("Personal details in your message were masked before processing: " + ", ".join(sorted(set(result["pii_found"]))))
        st.markdown(result["answer"])
        msg = {
            "role": "assistant", "content": result["answer"], "citations": result.get("citations", []),
            "route": result.get("route"), "sql": result.get("sql"), "trace": result.get("trace"),
            "latency_ms": result.get("latency_ms"),
        }
        render_extras(msg)
    st.session_state.messages.append(msg)
