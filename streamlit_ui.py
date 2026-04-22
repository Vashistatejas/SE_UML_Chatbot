import html

import streamlit as st
from dotenv import load_dotenv

from chatbot_core import (
    ModelConfig,
    build_api_messages,
    build_search_query,
    chunk_document_text,
    create_client,
    format_web_context,
    extract_text_from_document,
    format_document_context,
    get_model_config,
    is_tiet_related_prompt,
    load_system_prompt,
    search_web,
    select_relevant_document_chunks,
    should_search_web,
)


def render_styles() -> None:
    st.markdown(
        """
<style>
    .stApp {
        background: #101114;
        color: #f5f5f0;
    }

    h1 {
        text-align: center;
        font-family: Inter, system-ui, sans-serif;
        font-weight: 700;
        color: #f5f5f0;
        margin-bottom: 1.5rem;
    }

    .stChatMessage {
        background-color: #202126;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #34363d;
    }

    .stChatInputContainer {
        padding-bottom: 1.5rem;
    }

    [data-testid="stSidebar"] {
        background: #17181c;
        border-right: 1px solid #2d3036;
    }

    .source-card {
        border: 1px solid #34363d;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        background: #181a1f;
    }
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_cached_client(base_url: str | None, api_key: str | None, model: str):
    return create_client(ModelConfig(base_url=base_url, api_key=api_key, api_key_name=None, model=model))


@st.cache_data(show_spinner=False)
def get_cached_system_prompt() -> str:
    return load_system_prompt()


def initialize_state() -> None:
    defaults = {
        "messages": [],
        "student_details": None,
        "show_input_form": False,
        "last_sources": [],
        "document_chunks": [],
        "document_ids": [],
        "document_names": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_student_details() -> None:
    st.session_state.student_details = {
        "name": st.session_state.input_name,
        "roll_no": st.session_state.input_roll_no,
        "branch": st.session_state.input_branch,
    }
    st.session_state.show_input_form = False


def render_sidebar(config) -> tuple[bool, bool, int, bool]:
    st.sidebar.header("Assistant Settings")
    st.sidebar.caption(f"Model: `{config.model}`")
    if config.base_url:
        st.sidebar.caption(f"Endpoint: `{config.base_url}`")

    web_search_enabled = st.sidebar.toggle(
        "Search web for current questions",
        value=True,
        help="Automatically retrieves current web context for prompts that look time-sensitive.",
    )
    force_web_search = st.sidebar.toggle(
        "Search every Thapar prompt",
        value=False,
        help="Use this when you want citations for all Thapar answers, even stable questions.",
    )
    max_sources = st.sidebar.slider("Sources", min_value=2, max_value=6, value=4)
    show_sources = st.sidebar.toggle("Show retrieved sources", value=True)

    if st.sidebar.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

    render_document_controls()

    st.sidebar.divider()
    st.sidebar.subheader("Student Profile")
    if st.session_state.student_details:
        details = st.session_state.student_details
        st.sidebar.text(f"Name: {details.get('name', '')}")
        st.sidebar.text(f"Roll No: {details.get('roll_no', '')}")
        st.sidebar.text(f"Branch: {details.get('branch', '')}")
        if st.sidebar.button("Edit profile", use_container_width=True):
            st.session_state.show_input_form = True
    else:
        if st.sidebar.button("Add profile", use_container_width=True):
            st.session_state.show_input_form = True

    if st.session_state.show_input_form:
        with st.sidebar.form("details_form"):
            current = st.session_state.student_details or {}
            st.text_input("Name", value=current.get("name", ""), key="input_name")
            st.text_input("Roll No", value=current.get("roll_no", ""), key="input_roll_no")
            st.text_input("Branch", value=current.get("branch", ""), key="input_branch")
            st.form_submit_button("Save", on_click=save_student_details)

    return web_search_enabled, force_web_search, max_sources, show_sources


def render_document_controls() -> None:
    st.sidebar.divider()
    st.sidebar.subheader("Forms & Docs")
    uploaded_files = st.sidebar.file_uploader(
        "Add source files",
        type=["pdf", "docx", "txt", "md", "csv", "json"],
        accept_multiple_files=True,
        help="Uploaded documents are searched locally and cited as [D1], [D2], etc.",
    )

    for uploaded_file in uploaded_files:
        file_id = f"{uploaded_file.name}:{uploaded_file.size}"
        if file_id in st.session_state.document_ids:
            continue

        try:
            text = extract_text_from_document(uploaded_file.name, uploaded_file.getvalue())
            chunks = chunk_document_text(uploaded_file.name, text)
        except Exception as error:
            st.sidebar.warning(f"Could not read {uploaded_file.name}: {error}")
            continue

        if not chunks:
            st.sidebar.warning(f"No extractable text found in {uploaded_file.name}.")
            continue

        st.session_state.document_ids.append(file_id)
        st.session_state.document_names.append(uploaded_file.name)
        st.session_state.document_chunks.extend(chunks)

    if st.session_state.document_names:
        st.sidebar.caption(
            f"{len(st.session_state.document_names)} document(s), "
            f"{len(st.session_state.document_chunks)} searchable chunk(s)."
        )
        for name in st.session_state.document_names[-5:]:
            st.sidebar.text(f"- {name}")

        if st.sidebar.button("Clear documents", use_container_width=True):
            st.session_state.document_chunks = []
            st.session_state.document_ids = []
            st.session_state.document_names = []
            st.rerun()
    else:
        st.sidebar.caption("No documents added yet.")


def render_sources(sources) -> None:
    if not sources:
        st.info("No web sources were retrieved for the last answer.")
        return

    for index, source in enumerate(sources, start=1):
        title = html.escape(source.title)
        url = html.escape(source.url)
        snippet = html.escape(source.snippet or source.text[:280])
        st.markdown(
            f"""
<div class="source-card">
    <strong>[{index}] {title}</strong><br>
    <a href="{url}" target="_blank">{url}</a><br>
    <span>{snippet}</span>
</div>
""",
            unsafe_allow_html=True,
        )


def stream_completion(client, *, model: str, messages: list[dict[str, str]], placeholder) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    full_response = ""
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            placeholder.markdown(full_response + "...")

    placeholder.markdown(full_response)
    return full_response


def run_app() -> None:
    load_dotenv()
    st.set_page_config(page_title="TIET Assistant Chatbot", page_icon="T", layout="centered")
    render_styles()
    initialize_state()

    config = get_model_config()
    try:
        client = get_cached_client(config.base_url, config.api_key, config.model)
    except RuntimeError as error:
        st.error(str(error))
        st.info("Create a `.env` file with `OPENAI_API_KEY` or `HUGGINGFACE_API_KEY`.")
        st.stop()

    web_search_enabled, force_web_search, max_sources, show_sources = render_sidebar(config)
    system_prompt = get_cached_system_prompt()

    st.title("TIET Assistant Chatbot")

    if not st.session_state.messages:
        st.markdown(
            "<div style='display:flex;justify-content:center;align-items:center;"
            "height:45vh;flex-direction:column;color:#9a9a90;'>"
            "<h3>How can I help you?</h3></div>",
            unsafe_allow_html=True,
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if show_sources and st.session_state.last_sources:
        with st.expander("Sources from last answer", expanded=False):
            render_sources(st.session_state.last_sources)

    prompt = st.chat_input("Type your message here...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    web_context = ""
    document_context = ""
    sources = []
    relevant_document_chunks = select_relevant_document_chunks(
        prompt,
        st.session_state.document_chunks,
    )
    document_context = format_document_context(relevant_document_chunks)
    search_needed = (
        web_search_enabled
        and is_tiet_related_prompt(prompt)
        and (force_web_search or should_search_web(prompt))
    )

    if search_needed:
        query = build_search_query(prompt)
        with st.status("Retrieving Thapar-only current context...", expanded=False):
            sources = search_web(query, max_results=max_sources)
            web_context = format_web_context(sources, query)
            if sources:
                st.write(f"Found {len(sources)} source(s).")
            else:
                st.write("No Thapar-related sources found. The assistant will answer with available local context.")

    st.session_state.last_sources = sources
    api_messages = build_api_messages(
        base_prompt=system_prompt,
        history=st.session_state.messages[:-1],
        user_prompt=prompt,
        student_details=st.session_state.student_details,
        web_context=web_context,
        document_context=document_context,
    )

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            full_response = stream_completion(
                client,
                model=config.model,
                messages=api_messages,
                placeholder=message_placeholder,
            )
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as error:
            st.error(f"An error occurred while generating the answer: {error}")

    if show_sources and sources:
        with st.expander("Sources used", expanded=True):
            render_sources(sources)


if __name__ == "__main__":
    run_app()
