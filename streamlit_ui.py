from __future__ import annotations

import hashlib
import html

import streamlit as st
from dotenv import load_dotenv

from llm_service import ModelConfig, create_client, get_model_config
from rag_data import RagChunk, chunk_uploaded_document, extract_text_from_document, load_dataset_chunks
from rag_index import FaissChunkStore, RetrievalHit, build_transient_faiss_store, detect_domains, get_embedder
from rag_prompt import build_clarification_prompt, build_messages, format_retrieved_chunks, is_vague_query


def render_styles() -> None:
    st.markdown(
        """
<style>
    .stApp {
        background: #ffffff;
        color: #111827;
    }

    h1 {
        color: #0f172a;
        font-family: "Segoe UI", sans-serif;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    .stChatMessage {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        background: #ffffff;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
    }

    .chunk-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.9rem;
        margin-bottom: 0.75rem;
        background: #f8fafc;
    }

    .chunk-meta {
        color: #475569;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }

    .hero-copy {
        color: #475569;
        max-width: 48rem;
        margin-bottom: 1rem;
    }
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_cached_client(base_url: str | None, api_key: str | None, model: str):
    return create_client(ModelConfig(base_url=base_url, api_key=api_key, api_key_name=None, model=model))


@st.cache_resource(show_spinner=False)
def get_cached_embedder():
    return get_embedder()


@st.cache_resource(show_spinner=False)
def get_cached_dataset_chunks() -> list[RagChunk]:
    return load_dataset_chunks()


@st.cache_resource(show_spinner=False)
def get_cached_main_store() -> FaissChunkStore:
    chunks = get_cached_dataset_chunks()
    embedder = get_cached_embedder()
    return FaissChunkStore.load_or_create(chunks=chunks, embedder=embedder)


def initialize_state() -> None:
    defaults = {
        "messages": [],
        "last_retrieved_chunks": [],
        "uploaded_doc_ids": [],
        "uploaded_doc_chunks": [],
        "uploaded_doc_store": None,
        "uploaded_doc_signature": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def compute_upload_signature(files) -> str:
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(str(file.size).encode("utf-8"))
    return digest.hexdigest()


def sync_uploaded_documents(uploaded_files) -> None:
    signature = compute_upload_signature(uploaded_files)
    if signature == st.session_state.uploaded_doc_signature:
        return

    st.session_state.uploaded_doc_signature = signature
    st.session_state.uploaded_doc_ids = []
    st.session_state.uploaded_doc_chunks = []
    st.session_state.uploaded_doc_store = None

    if not uploaded_files:
        return

    next_index = 1
    for uploaded_file in uploaded_files:
        file_id = f"{uploaded_file.name}:{uploaded_file.size}"
        try:
            text = extract_text_from_document(uploaded_file.name, uploaded_file.getvalue())
            chunks = chunk_uploaded_document(
                filename=uploaded_file.name,
                text=text,
                start_index=next_index,
            )
        except Exception as error:
            st.sidebar.warning(f"Could not index {uploaded_file.name}: {error}")
            continue

        if not chunks:
            continue

        st.session_state.uploaded_doc_ids.append(file_id)
        st.session_state.uploaded_doc_chunks.extend(chunks)
        next_index += len(chunks)

    if st.session_state.uploaded_doc_chunks:
        embedder = get_cached_embedder()
        st.session_state.uploaded_doc_store = build_transient_faiss_store(
            chunks=st.session_state.uploaded_doc_chunks,
            embedder=embedder,
        )


def render_sidebar(config: ModelConfig) -> tuple[int, bool]:
    dataset_chunks = get_cached_dataset_chunks()
    embedder = get_cached_embedder()

    st.sidebar.header("RAG Settings")
    st.sidebar.caption(f"Chat model: `{config.model}`")
    st.sidebar.caption(f"Embedding: `{embedder.config.model}`")
    st.sidebar.caption(f"Dataset chunks: `{len(dataset_chunks)}`")

    top_k = st.sidebar.slider("Top-k retrieval", min_value=3, max_value=8, value=5)
    show_debug = st.sidebar.toggle("Show retrieved chunks", value=True)

    st.sidebar.divider()
    st.sidebar.subheader("Optional Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload extra PDFs or notes",
        type=["pdf", "docx", "txt", "md", "csv", "json"],
        accept_multiple_files=True,
    )
    sync_uploaded_documents(uploaded_files)

    if st.session_state.uploaded_doc_chunks:
        st.sidebar.caption(
            f"Uploaded document chunks: `{len(st.session_state.uploaded_doc_chunks)}`"
        )
        if st.sidebar.button("Clear uploaded documents", use_container_width=True):
            st.session_state.uploaded_doc_signature = ""
            st.session_state.uploaded_doc_ids = []
            st.session_state.uploaded_doc_chunks = []
            st.session_state.uploaded_doc_store = None
            st.rerun()
    else:
        st.sidebar.caption("No additional documents indexed.")

    if st.sidebar.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_retrieved_chunks = []
        st.rerun()

    return top_k, show_debug


def merge_hits(*hit_groups: list[RetrievalHit], top_k: int, max_context_tokens: int = 1000) -> list[RetrievalHit]:
    combined = [hit for group in hit_groups for hit in group]
    combined.sort(key=lambda hit: hit.score, reverse=True)

    selected: list[RetrievalHit] = []
    seen = set()
    total_tokens = 0

    for hit in combined:
        fingerprint = (
            hit.chunk.domain,
            hit.chunk.title.strip().lower(),
            hit.chunk.text.strip().lower(),
        )
        if fingerprint in seen:
            continue

        chunk_tokens = max(1, len(hit.chunk.text) // 4)
        if selected and total_tokens + chunk_tokens > max_context_tokens:
            continue

        selected.append(hit)
        seen.add(fingerprint)
        total_tokens += chunk_tokens
        if len(selected) >= top_k:
            break

    return selected


def render_retrieved_chunks(chunk_cards: list[dict[str, str]]) -> None:
    if not chunk_cards:
        st.info("No chunks were retrieved for the last answer.")
        return

    for card in chunk_cards:
        st.markdown(
            f"""
<div class="chunk-card">
    <strong>[{html.escape(card["chunk_id"])}] {html.escape(card["title"])}</strong>
    <div class="chunk-meta">
        {html.escape(card["domain"])} | {html.escape(card["section"])} | score {html.escape(card["score"])}
    </div>
    <div class="chunk-meta">{html.escape(card["source"])}</div>
    <pre style="white-space:pre-wrap;margin:0;">{html.escape(card["text"][:1000])}</pre>
</div>
""",
            unsafe_allow_html=True,
        )


def stream_completion(client, *, model: str, messages: list[dict[str, str]], placeholder) -> str:
    completion = client.chat.completions.create(model=model, messages=messages, stream=True)

    full_response = ""
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            placeholder.markdown(full_response + "▌")

    placeholder.markdown(full_response)
    return full_response


def retrieve_semantic_context(query: str, *, top_k: int) -> list[RetrievalHit]:
    main_store = get_cached_main_store()
    preferred_domains = detect_domains(query)
    dataset_hits = main_store.search(
        query,
        top_k=top_k,
        max_context_tokens=1000,
        preferred_domains=preferred_domains,
    )

    if st.session_state.uploaded_doc_store is None:
        return dataset_hits

    document_hits = st.session_state.uploaded_doc_store.search(
        query,
        top_k=min(2, top_k),
        max_context_tokens=350,
        preferred_domains={"document"} if "document" in preferred_domains else set(),
    )
    return merge_hits(dataset_hits, document_hits, top_k=top_k)


def run_app() -> None:
    load_dotenv()
    st.set_page_config(page_title="TIET RAG Assistant", page_icon="T", layout="centered")
    render_styles()
    initialize_state()

    try:
        config = get_model_config()
        client = get_cached_client(config.base_url, config.api_key, config.model)
        get_cached_main_store()
    except RuntimeError as error:
        st.error(str(error))
        st.info(
            "Set a chat API key and install `faiss-cpu`, `numpy`, and either "
            "`sentence-transformers` or `OPENAI_API_KEY` for embeddings."
        )
        st.stop()

    top_k, show_debug = render_sidebar(config)

    st.title("TIET Student Assistant")
    st.markdown(
        "<p class='hero-copy'>Semantic RAG over TIET hostel, faculty, club, and staff data. "
        "Only retrieved chunks are sent to the model.</p>",
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if show_debug and st.session_state.last_retrieved_chunks:
        with st.expander("Retrieved chunks from last answer", expanded=False):
            render_retrieved_chunks(st.session_state.last_retrieved_chunks)

    prompt = st.chat_input("Ask about hostels, faculty, research areas, or contacts")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if is_vague_query(prompt):
        clarification = build_clarification_prompt(prompt)
        st.session_state.last_retrieved_chunks = []
        with st.chat_message("assistant"):
            st.markdown(clarification)
        st.session_state.messages.append({"role": "assistant", "content": clarification})
        return

    with st.spinner("Retrieving relevant TIET chunks..."):
        hits = retrieve_semantic_context(prompt, top_k=top_k)
        retrieved_cards = format_retrieved_chunks(hits)
        st.session_state.last_retrieved_chunks = retrieved_cards

    if not hits:
        fallback = "I don't have that information."
        with st.chat_message("assistant"):
            st.markdown(fallback)
        st.session_state.messages.append({"role": "assistant", "content": fallback})
        return

    messages = build_messages(
        user_query=prompt,
        hits=hits,
        history=st.session_state.messages[:-1],
    )

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Generating answer..."):
            try:
                answer = stream_completion(
                    client,
                    model=config.model,
                    messages=messages,
                    placeholder=message_placeholder,
                )
            except Exception as error:
                st.error(f"Failed to generate answer: {error}")
                return

    st.session_state.messages.append({"role": "assistant", "content": answer})

    if show_debug:
        with st.expander("Retrieved chunks used", expanded=True):
            render_retrieved_chunks(retrieved_cards)


if __name__ == "__main__":
    run_app()
