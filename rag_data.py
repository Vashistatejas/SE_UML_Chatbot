from __future__ import annotations

import html
import io
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    source: str
    domain: str
    title: str
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RagChunk":
        return cls(
            chunk_id=payload["chunk_id"],
            source=payload["source"],
            domain=payload["domain"],
            title=payload["title"],
            text=payload["text"],
            metadata=payload.get("metadata", {}),
        )


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def normalize_label(value: str) -> str:
    value = normalize_whitespace(value.replace("_", " ").replace("-", " "))
    return value.title() if value else ""


def format_scalar(value: Any) -> str:
    if value is None:
        return "Not provided"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return normalize_whitespace(str(value))


def decode_text_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_text_from_document(filename: str, data: bytes) -> str:
    extension = Path(filename).suffix.lower()

    if extension in {".txt", ".md", ".csv", ".json"}:
        return normalize_whitespace(decode_text_bytes(data))

    if extension == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("Install pypdf to extract text from PDF files.") from error

        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"Page {page_number}: {text}")
        return normalize_whitespace("\n\n".join(pages))

    if extension == ".docx":
        return extract_docx_text(data)

    raise ValueError(
        "Unsupported file type. Upload PDF, DOCX, TXT, MD, CSV, or JSON files."
    )


def extract_docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        xml_data = archive.read("word/document.xml")

    root = ElementTree.fromstring(xml_data)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []

    for paragraph in root.findall(".//w:p", namespace):
        text_parts = [
            text_node.text or ""
            for text_node in paragraph.findall(".//w:t", namespace)
        ]
        paragraph_text = normalize_whitespace("".join(text_parts))
        if paragraph_text:
            paragraphs.append(paragraph_text)

    return normalize_whitespace("\n".join(paragraphs))


def chunk_uploaded_document(
    *,
    filename: str,
    text: str,
    start_index: int,
    max_chars: int = 1200,
    overlap: int = 160,
) -> list[RagChunk]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []

    chunks: list[RagChunk] = []
    cursor = 0
    chunk_number = 1
    next_id = start_index

    while cursor < len(normalized):
        end = min(cursor + max_chars, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", cursor, end)
            if boundary > cursor + max_chars // 2:
                end = boundary

        chunk_text = normalized[cursor:end].strip()
        if chunk_text:
            title = f"{filename} - Segment {chunk_number}"
            chunks.append(
                RagChunk(
                    chunk_id=f"U{next_id}",
                    source=filename,
                    domain="document",
                    title=title,
                    text=chunk_text,
                    metadata={
                        "section": "Uploaded Document",
                        "segment": chunk_number,
                        "source_type": "uploaded_document",
                    },
                )
            )
            next_id += 1
            chunk_number += 1

        if end >= len(normalized):
            break
        cursor = max(end - overlap, 0)

    return chunks


SECTION_HINT_MAP = {
    "hostel": "hostel",
    "hall": "hostel",
    "boys_hostels": "hostel",
    "girls_hostels": "hostel",
    "faculty": "faculty",
    "professor": "faculty",
    "mentor": "faculty",
    "staff": "staff",
    "club": "club",
    "society": "club",
    "chapter": "club",
}


def is_heading_candidate(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if set(line) == {"_"}:
        return False
    lowered = line.lower()
    if lowered.startswith(("user:", "assistant:", "example:")):
        return False
    if '"' in line:
        return False
    if line.startswith(("{", "}", '"', "[", "]")):
        return False
    if len(line) > 110:
        return False
    if ":" in line and line.count(":") > 1:
        return False
    return any(character.isalpha() for character in line)


def extract_json_sections(raw_text: str) -> list[tuple[str, Any]]:
    sections: list[tuple[str, Any]] = []
    decoder = json.JSONDecoder()
    search_start = 0

    while True:
        start_index = raw_text.find("{", search_start)
        if start_index < 0:
            break

        heading = ""
        prefix_lines = raw_text[:start_index].splitlines()
        for candidate in reversed(prefix_lines[-8:]):
            if is_heading_candidate(candidate):
                heading = normalize_whitespace(candidate)
                break

        try:
            parsed, consumed = decoder.raw_decode(raw_text[start_index:])
        except json.JSONDecodeError:
            search_start = start_index + 1
            continue

        sections.append((heading, parsed))
        search_start = start_index + consumed

    return sections


def infer_domain(section_label: str, data: Any) -> str:
    lowered_label = section_label.lower()
    for hint, domain in SECTION_HINT_MAP.items():
        if hint in lowered_label:
            return domain

    if isinstance(data, dict):
        keys = {key.lower() for key in data}
        if {"boys_hostels", "girls_hostels", "common_facilities"} & keys:
            return "hostel"
        if "faculty" in keys:
            return "faculty"
        if "tiet_staff" in keys:
            return "staff"
        if all(
            isinstance(value, dict) and {"details", "email"} & set(value.keys())
            for value in data.values()
        ):
            return "club"

    return "general"


def default_section_for_domain(domain: str) -> str:
    return {
        "faculty": "Faculty Directory",
        "hostel": "Hostel Directory",
        "staff": "TIET Staff Directory",
        "club": "Clubs And Societies",
        "general": "TIET Knowledge Base",
    }.get(domain, "TIET Knowledge Base")


def canonicalize_year(value: Any) -> str:
    text = format_scalar(value)
    return text.replace("_", " ").title()


def build_hostel_aliases(record: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    code = format_scalar(record.get("code"))
    if code and code != "Not provided":
        aliases.extend([f"Hostel {code}", f"Hall {code}", f"{code} Hall"])
    return aliases


def format_field_lines(key: str, value: Any, *, indent: int = 0) -> list[str]:
    prefix = " " * indent
    label = normalize_label(key)

    if isinstance(value, dict):
        lines = [f"{prefix}{label}:"]
        for nested_key, nested_value in value.items():
            lines.extend(format_field_lines(nested_key, nested_value, indent=indent + 2))
        return lines

    if isinstance(value, list):
        if not value:
            return [f"{prefix}{label}: Not provided"]
        if all(not isinstance(item, (dict, list)) for item in value):
            return [f"{prefix}{label}: {', '.join(format_scalar(item) for item in value)}"]

        lines = [f"{prefix}{label}:"]
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}  -")
                for nested_key, nested_value in item.items():
                    lines.extend(
                        format_field_lines(nested_key, nested_value, indent=indent + 4)
                    )
            else:
                lines.append(f"{prefix}  - {format_scalar(item)}")
        return lines

    return [f"{prefix}{label}: {format_scalar(value)}"]


def build_chunk_text(
    *,
    domain: str,
    section: str,
    title: str,
    record: dict[str, Any],
    inherited_fields: dict[str, Any] | None = None,
) -> str:
    lines = [
        f"Domain: {normalize_label(domain)}",
        f"Section: {section}",
        f"Title: {title}",
    ]

    if domain == "hostel":
        aliases = build_hostel_aliases(record)
        if aliases:
            lines.append(f"Aliases: {', '.join(aliases)}")

    if inherited_fields:
        for key, value in inherited_fields.items():
            lines.extend(format_field_lines(key, value))

    for key, value in record.items():
        if key == "year":
            lines.append(f"Year: {canonicalize_year(value)}")
            continue
        lines.extend(format_field_lines(key, value))

    return "\n".join(line for line in lines if line.strip())


def create_chunk(
    *,
    next_index: int,
    source: str,
    domain: str,
    section: str,
    title: str,
    record: dict[str, Any],
    inherited_fields: dict[str, Any] | None = None,
) -> RagChunk:
    metadata = {
        "section": section,
        "source_type": "tiet_dataset",
        "title": title,
    }
    if domain == "hostel" and record.get("code") is not None:
        metadata["code"] = format_scalar(record.get("code"))

    return RagChunk(
        chunk_id=f"K{next_index}",
        source=source,
        domain=domain,
        title=title,
        text=build_chunk_text(
            domain=domain,
            section=section,
            title=title,
            record=record,
            inherited_fields=inherited_fields,
        ),
        metadata=metadata,
    )


def chunk_json_section(
    data: Any,
    *,
    section_label: str,
    source: str,
    start_index: int,
) -> list[RagChunk]:
    domain = infer_domain(section_label, data)
    normalized_label = section_label.lower()
    inferred_label = default_section_for_domain(domain)
    if not section_label or (
        domain not in normalized_label
        and not any(
            hint in normalized_label
            for hint, mapped_domain in SECTION_HINT_MAP.items()
            if mapped_domain == domain
        )
    ):
        section = inferred_label
    else:
        section = section_label

    next_index = start_index
    chunks: list[RagChunk] = []

    if isinstance(data, dict) and "faculty" in data and isinstance(data["faculty"], list):
        inherited = {key: value for key, value in data.items() if key != "faculty"}
        for faculty in data["faculty"]:
            if not isinstance(faculty, dict):
                continue
            title = faculty.get("name") or f"Faculty {next_index}"
            chunks.append(
                create_chunk(
                    next_index=next_index,
                    source=source,
                    domain="faculty",
                    section=section,
                    title=title,
                    record=faculty,
                    inherited_fields=inherited or None,
                )
            )
            next_index += 1
        return chunks

    if isinstance(data, dict) and "tiet_staff" in {key.lower() for key in data}:
        staff_members = next(
            (
                value
                for key, value in data.items()
                if key.lower() == "tiet_staff" and isinstance(value, list)
            ),
            [],
        )
        for staff_member in staff_members:
            if not isinstance(staff_member, dict):
                continue
            title = (
                staff_member.get("name")
                or staff_member.get("role")
                or f"Staff {next_index}"
            )
            chunks.append(
                create_chunk(
                    next_index=next_index,
                    source=source,
                    domain="staff",
                    section=section,
                    title=title,
                    record=staff_member,
                )
            )
            next_index += 1
        return chunks

    if isinstance(data, dict) and any(
        key in data for key in ("boys_hostels", "girls_hostels", "common_facilities")
    ):
        common_facilities = data.get("common_facilities")
        if isinstance(common_facilities, dict):
            chunks.append(
                create_chunk(
                    next_index=next_index,
                    source=source,
                    domain="hostel",
                    section=section,
                    title="Common Hostel Facilities",
                    record=common_facilities,
                )
            )
            next_index += 1

        for hostel_group in ("boys_hostels", "girls_hostels"):
            hostels = data.get(hostel_group)
            if not isinstance(hostels, dict):
                continue
            for hostel_name, hostel_payload in hostels.items():
                if not isinstance(hostel_payload, dict):
                    continue
                chunks.append(
                    create_chunk(
                        next_index=next_index,
                        source=source,
                        domain="hostel",
                        section=f"{section} / {normalize_label(hostel_group)}",
                        title=hostel_name,
                        record=hostel_payload,
                    )
                )
                next_index += 1
        return chunks

    if isinstance(data, dict) and all(isinstance(value, dict) for value in data.values()):
        for item_name, item_payload in data.items():
            chunks.append(
                create_chunk(
                    next_index=next_index,
                    source=source,
                    domain=domain,
                    section=section,
                    title=item_name,
                    record=item_payload,
                )
            )
            next_index += 1
        return chunks

    if isinstance(data, list):
        for item_payload in data:
            if not isinstance(item_payload, dict):
                continue
            title = item_payload.get("name") or item_payload.get("title") or f"Item {next_index}"
            chunks.append(
                create_chunk(
                    next_index=next_index,
                    source=source,
                    domain=domain,
                    section=section,
                    title=title,
                    record=item_payload,
                )
            )
            next_index += 1
        return chunks

    if isinstance(data, dict):
        chunks.append(
            create_chunk(
                next_index=next_index,
                source=source,
                domain=domain,
                section=section,
                title=section,
                record=data,
            )
        )

    return chunks


def load_dataset_chunks(path: str = "llm_main_prompt.txt") -> list[RagChunk]:
    raw_text = Path(path).read_text(encoding="utf-8")
    chunks: list[RagChunk] = []
    next_index = 1

    for section_label, payload in extract_json_sections(raw_text):
        new_chunks = chunk_json_section(
            payload,
            section_label=section_label,
            source=path,
            start_index=next_index,
        )
        chunks.extend(new_chunks)
        next_index += len(new_chunks)

    return chunks
