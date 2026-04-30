from __future__ import annotations

import re

from rag_data import RagChunk
from rag_index import RetrievalHit


SYSTEM_PROMPT = (
    "You are a TIET assistant. Answer ONLY using the provided context. "
    'If answer is not found, say: "I don\'t have that information."'
)

VAGUE_QUERY_PATTERNS = (
    r"^\s*(hostel|hostels|faculty|professor|professors|fees|fee|warden|research)\s*$",
    r"^\s*(tell me|help me|information)\s*$",
)


def is_vague_query(query: str) -> bool:
    stripped = query.strip()
    if len(stripped.split()) <= 1:
        return True
    return any(re.match(pattern, stripped, flags=re.IGNORECASE) for pattern in VAGUE_QUERY_PATTERNS)


def build_clarification_prompt(query: str) -> str:
    lowered = query.lower()
    if "hostel" in lowered:
        return "Please specify the hostel name, hostel code, or the exact hostel detail you need."
    if any(term in lowered for term in ("faculty", "professor", "mentor", "research")):
        return "Please specify the faculty name, department, subject, or research area you want to know about."
    return "Please clarify whether you need hostel information or faculty information, and include the specific detail you want."


def format_context_block(hits: list[RetrievalHit]) -> str:
    sections = []
    for hit in hits:
        chunk = hit.chunk
        metadata_bits = [chunk.domain]
        section = chunk.metadata.get("section")
        if section:
            metadata_bits.append(str(section))
        sections.append(
            "\n".join(
                [
                    f"[{chunk.chunk_id}] {chunk.title}",
                    f"Source: {chunk.source}",
                    f"Metadata: {' | '.join(metadata_bits)}",
                    chunk.text,
                ]
            )
        )
    return "\n\n".join(sections)


def build_messages(
    *,
    user_query: str,
    hits: list[RetrievalHit],
    history: list[dict[str, str]],
    max_history_messages: int = 6,
) -> list[dict[str, str]]:
    recent_history = [
        {"role": message["role"], "content": message["content"]}
        for message in history[-max_history_messages:]
        if message.get("role") in {"user", "assistant"} and message.get("content")
    ]

    user_message = "\n".join(
        [
            "CONTEXT:",
            format_context_block(hits) or "No relevant context retrieved.",
            "",
            "QUESTION:",
            user_query,
            "",
            "INSTRUCTIONS:",
            "Be precise and structured.",
            "Use bullet points when helpful.",
            "Do NOT hallucinate.",
            "Mention chunk IDs used.",
            "If multiple chunks conflict, mention uncertainty.",
        ]
    )

    return [{"role": "system", "content": SYSTEM_PROMPT}] + recent_history + [
        {"role": "user", "content": user_message}
    ]


def format_retrieved_chunks(hits: list[RetrievalHit]) -> list[dict[str, str]]:
    return [
        {
            "chunk_id": hit.chunk.chunk_id,
            "title": hit.chunk.title,
            "domain": hit.chunk.domain,
            "source": hit.chunk.source,
            "section": str(hit.chunk.metadata.get("section", "")),
            "text": hit.chunk.text,
            "score": f"{hit.score:.4f}",
        }
        for hit in hits
    ]
