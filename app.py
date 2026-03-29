import os
import shutil

import streamlit as st

from rag_utility import process_documents_to_chroma_db, answer_question

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

# ── Minimal custom CSS ────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .source-badge {
            display: inline-block;
            background: #1e3a5f;
            color: #7ec8e3;
            border: 1px solid #2a5298;
            border-radius: 4px;
            padding: 2px 10px;
            margin: 3px 4px;
            font-size: 0.82rem;
            font-family: monospace;
        }
        .answer-box {
            background: #0f1e2e;
            border-left: 4px solid #2a5298;
            border-radius: 6px;
            padding: 1rem 1.2rem;
            margin-top: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Working directory ─────────────────────────────────────────────────────────
working_dir = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_DIR = os.path.join(working_dir, "doc_vectorstore")

# ── Session state ─────────────────────────────────────────────────────────────
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []   # list of file names already embedded
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []       # list of {question, answer, sources}

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📚 Multi-PDF RAG Assistant")
st.caption("Upload several PDFs, then ask questions across all of them.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Document Manager")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDF files at once.",
    )

    process_btn = st.button("⚙️ Process / Re-index Documents", use_container_width=True)

    if process_btn:
        if not uploaded_files:
            st.warning("Please upload at least one PDF before processing.")
        else:
            saved_names = []
            for uf in uploaded_files:
                save_path = os.path.join(working_dir, uf.name)
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())
                saved_names.append(uf.name)

            with st.spinner("Embedding documents — this may take a minute…"):
                try:
                    # Wipe old vector store so we start fresh each time
                    if os.path.exists(VECTORSTORE_DIR):
                        shutil.rmtree(VECTORSTORE_DIR)

                    total_chunks = process_documents_to_chroma_db(saved_names)
                    st.session_state.processed_files = saved_names
                    st.session_state.chat_history = []   # clear old chat on re-index
                    st.success(
                        f"✅ Indexed **{len(saved_names)}** file(s) "
                        f"into **{total_chunks}** chunks."
                    )
                except Exception as e:
                    st.error(f"Error during processing: {e}")

    # Show currently indexed files
    if st.session_state.processed_files:
        st.divider()
        st.subheader("Indexed Files")
        for fname in st.session_state.processed_files:
            st.markdown(f"- 📄 `{fname}`")

        if st.button("🗑️ Clear Index & History", use_container_width=True):
            if os.path.exists(VECTORSTORE_DIR):
                shutil.rmtree(VECTORSTORE_DIR)
            st.session_state.processed_files = []
            st.session_state.chat_history = []
            st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────
if not st.session_state.processed_files:
    st.info("👈 Upload your PDFs in the sidebar and click **Process / Re-index Documents** to get started.")
else:
    # Display chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(f'<div class="answer-box">{entry["answer"]}</div>', unsafe_allow_html=True)
            if entry["sources"]:
                st.markdown("**Sources:**")
                badges = "".join(
                    f'<span class="source-badge">📄 {s}</span>'
                    for s in entry["sources"]
                )
                st.markdown(badges, unsafe_allow_html=True)

    # Input box at the bottom
    user_question = st.chat_input("Ask a question across your documents…")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer…"):
                try:
                    answer, sources = answer_question(user_question)
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
                    sources = []

            st.markdown(
                f'<div class="answer-box">{answer}</div>',
                unsafe_allow_html=True,
            )
            if sources:
                st.markdown("**Sources:**")
                badges = "".join(
                    f'<span class="source-badge">📄 {s}</span>'
                    for s in sources
                )
                st.markdown(badges, unsafe_allow_html=True)

        # Save to history
        st.session_state.chat_history.append(
            {"question": user_question, "answer": answer, "sources": sources}
        )