# TIET Assistant Chatbot

An interactive Streamlit chatbot for Thapar Institute of Engineering and Technology (TIET). The app uses the local TIET prompt dataset for stable university facts, can retrieve Thapar-related live web context for current questions, and can answer from uploaded forms or documents.

## What Changed

- **Vector-Based RAG System**: Replaced monolithic prompts with semantic search using FAISS and embeddings (OpenAI or `sentence-transformers`).
- **Smart Chunking & Extraction**: Parses JSON and text documents (`rag_data.py`) into metadata-rich `RagChunk` objects for precise context retrieval.
- **Local Index Caching**: Vector indices are cached locally in `.cache/tiet_rag` for faster application restarts.
- Adds automatic web retrieval for time-sensitive prompts.
- Filters web results to Thapar/TIET-related sources and ranks official `thapar.edu` pages first.
- Blocks irrelevant source domains such as Slack and social pages.
- Passes retrieved source context into the model and asks it to cite sources with `[1]`, `[2]`, etc.
- Adds document upload for PDF, DOCX, TXT, MD, CSV, and JSON files. Extracted document context is cited as `[D1]`, `[D2]`, etc.
- Keeps the local `llm_main_prompt.txt` dataset for stable TIET answers, now dynamically chunked and indexed.
- Supports Groq, Hugging Face Router, OpenAI, or any OpenAI-compatible endpoint through environment variables.
- Consolidates duplicate app logic into `streamlit_ui.py` so `streamlit_app.py` and `final.py` run the same code.
- Adds sidebar controls for web search, source count, source display, chat clearing, and student profile details.

## Project Structure

- `streamlit_app.py`: Main Streamlit entrypoint.
- `final.py`: Compatibility entrypoint that launches the same app.
- `streamlit_ui.py`: Streamlit UI, chat flow, sidebar controls, and streaming response handling.
- `chatbot_core.py`: Model configuration, prompt construction, Thapar-only web search, and message preparation.
- `rag_data.py`: Handles data chunking, hierarchical extraction from JSON files, and document parsing.
- `rag_index.py`: Implements vector search using FAISS, supporting multiple embedding providers (OpenAI, sentence-transformers).
- `rag_prompt.py`: Manages system prompt construction, context formatting, and query clarification logic.
- `llm_main_prompt.txt`: Local TIET dataset dynamically chunked into the vector store.
- `requirements.txt`: Python dependencies.
- `.env.example`: Example environment configuration.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example` and add an API key.

For Groq GPT-OSS 120B:

```env
GROQ_API_KEY=your_groq_api_key
CHATBOT_MODEL=openai/gpt-oss-120b
```

The app automatically uses Groq's OpenAI-compatible endpoint:

```env
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

You only need to set `GROQ_BASE_URL` if you want to override the endpoint explicitly.

For OpenAI instead:

```env
OPENAI_API_KEY=your_openai_api_key
CHATBOT_MODEL=gpt-5.4-mini
```

For Hugging Face Router instead:

```env
HUGGINGFACE_API_KEY=your_huggingface_api_key
CHATBOT_MODEL=openai/gpt-oss-120b
```

For another OpenAI-compatible provider:

```env
OPENAI_API_KEY=your_provider_key
OPENAI_BASE_URL=https://your-provider.example/v1
CHATBOT_MODEL=your-model-name
```

**RAG / Embeddings Configuration:**

By default, the app uses `openai` if `OPENAI_API_KEY` is present, or falls back to local `sentence-transformers`. You can explicitly configure the provider:

```env
EMBEDDING_PROVIDER=openai # or sentence-transformers, huggingface
# Optional: Override the default models
# EMBEDDING_MODEL=text-embedding-3-small
```

4. Run the app:

```bash
streamlit run streamlit_app.py
```

## How Fresh Answers Work

The app checks each prompt for time-sensitive words such as `latest`, `current`, `today`, `admission`, `fee`, `deadline`, `notice`, `event`, `schedule`, and `placement`. It only searches when the prompt is also related to Thapar/TIET. The search query is expanded with Thapar context, official `thapar.edu` sources are ranked highest, and unrelated domains such as Slack/social pages are removed before the model sees the evidence.

Use the sidebar if you want to:

- turn web retrieval on or off;
- force search for every Thapar prompt;
- change how many sources are retrieved;
- show or hide retrieved sources.

The local prompt still matters for stable TIET facts. The web context is used when information may have changed.

## Uploaded Forms And Documents

Use the sidebar's **Forms & Docs** uploader to add PDF, DOCX, TXT, MD, CSV, or JSON files. The app extracts text, breaks it into searchable chunks using `rag_data.py`, and adds it to a dynamic FAISS vector index using `rag_index.py`. It then includes only the most semantically relevant chunks in the model request.

This is useful for:

- hostel fee circulars;
- admission brochures;
- counselling PDFs;
- event notices;
- academic forms;
- placement or department documents.

Scanned image-only PDFs need OCR before upload because regular PDF text extraction cannot read text that exists only as an image.

## Model Recommendation

For your requested Groq setup, use:

```env
GROQ_API_KEY=your_groq_api_key
CHATBOT_MODEL=openai/gpt-oss-120b
```

`openai/gpt-oss-120b` is the better Groq choice for this project than the smaller 20B model because it should follow retrieval/document grounding instructions more reliably. If you later prioritize maximum accuracy over Groq latency/cost, move to a stronger OpenAI model such as `gpt-5.4` or `gpt-5.4-mini`.
