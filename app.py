import os
import streamlit as st
from chunking import make_chunks
from retrieval import build_db, retrieve, ask_llm

st.title("Milestone 7 RAG")
if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "db" not in st.session_state:
    st.session_state.db = None
if "ready" not in st.session_state:
    st.session_state.ready = False

with st.sidebar:
    key = st.text_input("GROQ_API_KEY", type="password")
    if key:
        os.environ["GROQ_API_KEY"] = key
    files = st.file_uploader(
        "Upload PDF/TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    if st.button("Index"):
        if files:
            chunks, metas, ids = make_chunks(files)
            db = build_db(chunks, metas, ids)
            st.session_state.db = db
            st.session_state.ready = True
            st.success(f"Indexed {len(chunks)} chunks")
        else:
            st.warning("Upload files first")

for m in st.session_state.msgs:
    with st.chat_message(m["r"]):
        st.write(m["c"])

q = st.chat_input("Ask...")

if q:
    st.session_state.msgs.append({"r": "user", "c": q})
    with st.chat_message("user"):
        st.write(q)
    if not st.session_state.ready:
        a = "Index first."
    else:
        hits = retrieve(st.session_state.db, q)
        ctx = ""
        src = []
        for d, m in hits:
            ctx += d + "\n\n"
            src.append(m["file"])
        ans = ask_llm(ctx, q)
        a = ans + "\n\nSources:\n" + "\n".join(set(src))
    st.session_state.msgs.append({"r": "assistant", "c": a})
    with st.chat_message("assistant"):
        st.write(a)
