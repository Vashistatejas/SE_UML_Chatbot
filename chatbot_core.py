from __future__ import annotations

import html
import io
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

from openai import OpenAI


DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_HF_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_HF_MODEL = "openai/gpt-oss-120b"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"

OFFICIAL_THAPAR_DOMAINS = (
    "thapar.edu",
    "www.thapar.edu",
    "admissions.thapar.edu",
)

BLOCKED_SOURCE_DOMAINS = (
    "slack.com",
    "app.slack.com",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "x.com",
    "twitter.com",
    "youtube.com",
    "reddit.com",
    "quora.com",
    "pinterest.com",
)

TIET_CONTEXT_TERMS = (
    "thapar",
    "tiet",
    "patiala",
    "doaa",
    "dosa",
    "hostel",
    "mess",
    "warden",
    "faculty",
    "professor",
    "mentor",
    "club",
    "society",
    "faps",
    "saturnalia",
    "aranya",
    "admission",
    "admissions",
    "cutoff",
    "cut-off",
    "jee",
    "coe",
    "copc",
    "cobs",
    "branch",
    "semester",
    "fee",
    "fees",
)

DOCUMENT_REFERENCE_TERMS = (
    "document",
    "documents",
    "doc",
    "docs",
    "form",
    "forms",
    "file",
    "files",
    "pdf",
    "uploaded",
    "attachment",
)

STOPWORDS = {
    "about",
    "after",
    "all",
    "also",
    "and",
    "are",
    "can",
    "for",
    "from",
    "have",
    "how",
    "into",
    "know",
    "more",
    "next",
    "should",
    "tell",
    "that",
    "the",
    "this",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
}

TIME_SENSITIVE_PATTERNS = (
    "latest",
    "current",
    "currently",
    "today",
    "tomorrow",
    "yesterday",
    "recent",
    "new",
    "news",
    "update",
    "updated",
    "deadline",
    "schedule",
    "date",
    "admission",
    "admissions",
    "fee",
    "fees",
    "cutoff",
    "cut-off",
    "placement",
    "rank",
    "ranking",
    "result",
    "notice",
    "circular",
    "event",
    "fest",
    "semester",
    "timetable",
    "exam",
    "web",
    "search",
    "verify",
)


@dataclass(frozen=True)
class ModelConfig:
    base_url: str | None
    api_key: str | None
    api_key_name: str | None
    model: str


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    text: str = ""


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    chunk_id: int
    text: str


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source: str
    domain: str
    section: str
    title: str
    text: str


class DuckDuckGoResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._current_title: list[str] | None = None
        self._current_href = ""
        self._current_snippet: list[str] | None = None
        self._capture_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {name: value or "" for name, value in attrs}
        classes = attrs_dict.get("class", "")

        if tag == "a" and "result__a" in classes:
            self._flush_result()
            self._current_title = []
            self._current_href = attrs_dict.get("href", "")
            self._capture_title = True
            return

        if "result__snippet" in classes:
            self._capture_title = False
            self._current_snippet = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture_title:
            self._capture_title = False
        if self._current_snippet is not None and tag in {"a", "div"}:
            self._flush_result()

    def handle_data(self, data: str) -> None:
        if self._current_title is not None and self._capture_title:
            self._current_title.append(data)
        if self._current_snippet is not None:
            self._current_snippet.append(data)

    def close(self) -> None:
        self._flush_result()
        super().close()

    def _flush_result(self) -> None:
        if self._current_title is None:
            return

        title = normalize_whitespace(" ".join(self._current_title))
        url = unwrap_duckduckgo_url(self._current_href)
        snippet = normalize_whitespace(" ".join(self._current_snippet or []))

        if title and url and url.startswith(("http://", "https://")):
            self.results.append(SearchResult(title=title, url=url, snippet=snippet))

        self._current_title = None
        self._current_href = ""
        self._current_snippet = None
        self._capture_title = False


class HTMLTextParser(HTMLParser):
    def __init__(self, max_chars: int) -> None:
        super().__init__()
        self.max_chars = max_chars
        self.text_parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        elif tag in {"p", "br", "li", "h1", "h2", "h3", "td", "th"}:
            self.text_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag in {"p", "li", "h1", "h2", "h3", "tr"}:
            self.text_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if sum(len(part) for part in self.text_parts) >= self.max_chars:
            return
        self.text_parts.append(data)

    def get_text(self) -> str:
        return normalize_whitespace(" ".join(self.text_parts))[: self.max_chars]


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def unwrap_duckduckgo_url(url: str) -> str:
    if not url:
        return ""

    parsed = urllib.parse.urlparse(html.unescape(url))
    query = urllib.parse.parse_qs(parsed.query)
    if "uddg" in query:
        return query["uddg"][0]
    if parsed.scheme:
        return urllib.parse.urlunparse(parsed)
    return urllib.parse.urljoin("https://duckduckgo.com", url)


def get_model_config() -> ModelConfig:
    explicit_base_url = os.environ.get("OPENAI_BASE_URL")
    groq_base_url = os.environ.get("GROQ_BASE_URL")
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    huggingface_key = os.environ.get("HUGGINGFACE_API_KEY")

    if groq_key:
        return ModelConfig(
            base_url=groq_base_url or DEFAULT_GROQ_BASE_URL,
            api_key=groq_key,
            api_key_name="GROQ_API_KEY",
            model=os.environ.get("CHATBOT_MODEL", DEFAULT_GROQ_MODEL),
        )

    if huggingface_key and not openai_key:
        return ModelConfig(
            base_url=explicit_base_url or DEFAULT_HF_BASE_URL,
            api_key=huggingface_key,
            api_key_name="HUGGINGFACE_API_KEY",
            model=os.environ.get("CHATBOT_MODEL", DEFAULT_HF_MODEL),
        )

    return ModelConfig(
        base_url=explicit_base_url,
        api_key=openai_key,
        api_key_name="OPENAI_API_KEY" if openai_key else None,
        model=os.environ.get("CHATBOT_MODEL", DEFAULT_OPENAI_MODEL),
    )


def create_client(config: ModelConfig | None = None) -> OpenAI:
    config = config or get_model_config()
    if not config.api_key:
        expected = "GROQ_API_KEY, OPENAI_API_KEY, or HUGGINGFACE_API_KEY"
        raise RuntimeError(f"Missing API key. Set {expected} in your .env file.")

    kwargs = {"api_key": config.api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs)


DEFAULT_RAG_SYSTEM_PROMPT = (
    "You are a TIET student assistant. Answer only using the provided context. "
    'If information is not present, say "I don\'t have that information." '
    "If the question is vague, ask a brief clarification question. "
    "If multiple retrieved chunks conflict, mention the uncertainty. "
    "Cite the chunk ids you used, such as [K1], [D1], or [W1]."
)

HOSTEL_QUERY_TERMS = {
    "hostel",
    "hostels",
    "mess",
    "warden",
    "wardens",
    "hall",
    "rooms",
    "room",
    "laundry",
    "canteen",
    "fees",
    "fee",
}

FACULTY_QUERY_TERMS = {
    "faculty",
    "professor",
    "professors",
    "teacher",
    "teachers",
    "mentor",
    "mentors",
    "research",
    "specialization",
    "department",
    "designation",
    "subject",
    "subjects",
}

CLUB_QUERY_TERMS = {
    "club",
    "clubs",
    "society",
    "societies",
    "chapter",
    "faps",
    "mudra",
    "saturnalia",
    "aranya",
}

STAFF_QUERY_TERMS = {
    "staff",
    "office",
    "dean",
    "dosa",
    "adosa",
    "admin",
    "contact",
}

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


def load_system_prompt(path: str = "llm_main_prompt.txt") -> str:
    return DEFAULT_RAG_SYSTEM_PROMPT


def load_local_knowledge_text(path: str = "llm_main_prompt.txt") -> str:
    try:
        with open(path, "r", encoding="utf-8") as prompt_file:
            return prompt_file.read()
    except FileNotFoundError:
        return ""


def normalize_label(value: str) -> str:
    value = normalize_whitespace(value.replace("_", " ").replace("-", " "))
    return value.title() if value else ""


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
    active_heading = ""

    while True:
        start_index = raw_text.find("{", search_start)
        if start_index < 0:
            break

        prefix_lines = raw_text[:start_index].splitlines()
        for candidate in reversed(prefix_lines[-8:]):
            if is_heading_candidate(candidate):
                active_heading = normalize_whitespace(candidate)
                break

        try:
            parsed, consumed = decoder.raw_decode(raw_text[start_index:])
        except json.JSONDecodeError:
            search_start = start_index + 1
            continue

        sections.append((active_heading, parsed))
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


def format_scalar(value: Any) -> str:
    if value is None:
        return "Not provided"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return normalize_whitespace(str(value))


def format_field_block(key: str, value: Any, *, indent: int = 0) -> list[str]:
    prefix = " " * indent
    label = normalize_label(key)

    if isinstance(value, dict):
        lines = [f"{prefix}{label}:"]
        for nested_key, nested_value in value.items():
            lines.extend(format_field_block(nested_key, nested_value, indent=indent + 2))
        return lines

    if isinstance(value, list):
        if not value:
            return [f"{prefix}{label}: Not provided"]
        if all(not isinstance(item, (dict, list)) for item in value):
            joined = ", ".join(format_scalar(item) for item in value)
            return [f"{prefix}{label}: {joined}"]

        lines = [f"{prefix}{label}:"]
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}  -")
                for nested_key, nested_value in item.items():
                    lines.extend(
                        format_field_block(nested_key, nested_value, indent=indent + 4)
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
        f"Section: {section or 'TIET Knowledge Base'}",
        f"Title: {title}",
    ]

    if inherited_fields:
        for key, value in inherited_fields.items():
            lines.extend(format_field_block(key, value))

    for key, value in record.items():
        lines.extend(format_field_block(key, value))

    return "\n".join(line for line in lines if line.strip())


def create_knowledge_chunk(
    *,
    counter: int,
    source: str,
    domain: str,
    section: str,
    title: str,
    record: dict[str, Any],
    inherited_fields: dict[str, Any] | None = None,
) -> KnowledgeChunk:
    text = build_chunk_text(
        domain=domain,
        section=section,
        title=title,
        record=record,
        inherited_fields=inherited_fields,
    )
    return KnowledgeChunk(
        chunk_id=f"K{counter}",
        source=source,
        domain=domain,
        section=section or "TIET Knowledge Base",
        title=title,
        text=text,
    )


def default_section_for_domain(domain: str) -> str:
    return {
        "faculty": "Faculty Directory",
        "hostel": "Hostel Directory",
        "staff": "TIET Staff Directory",
        "club": "Clubs And Societies",
    }.get(domain, "TIET Knowledge Base")


def chunk_json_section(
    data: Any,
    *,
    section_label: str,
    source: str,
    start_counter: int,
) -> list[KnowledgeChunk]:
    domain = infer_domain(section_label, data)
    section_label_lower = section_label.lower()
    domain_label_is_misaligned = domain not in section_label_lower and not any(
        hint in section_label_lower
        for hint, mapped_domain in SECTION_HINT_MAP.items()
        if mapped_domain == domain
    )
    section = (
        section_label
        if section_label and not domain_label_is_misaligned
        else default_section_for_domain(domain)
    )
    counter = start_counter
    chunks: list[KnowledgeChunk] = []

    if isinstance(data, dict) and "faculty" in data and isinstance(data["faculty"], list):
        inherited = {key: value for key, value in data.items() if key != "faculty"}
        for faculty in data["faculty"]:
            if not isinstance(faculty, dict):
                continue
            title = faculty.get("name") or f"Faculty {counter}"
            chunks.append(
                create_knowledge_chunk(
                    counter=counter,
                    source=source,
                    domain="faculty",
                    section=section,
                    title=title,
                    record=faculty,
                    inherited_fields=inherited or None,
                )
            )
            counter += 1
        return chunks

    if isinstance(data, dict) and "tiet_staff" in {key.lower() for key in data}:
        staff_items = next(
            (
                value
                for key, value in data.items()
                if key.lower() == "tiet_staff" and isinstance(value, list)
            ),
            [],
        )
        for staff in staff_items:
            if not isinstance(staff, dict):
                continue
            title = staff.get("name") or staff.get("role") or f"Staff {counter}"
            chunks.append(
                create_knowledge_chunk(
                    counter=counter,
                    source=source,
                    domain="staff",
                    section=section,
                    title=title,
                    record=staff,
                )
            )
            counter += 1
        return chunks

    if isinstance(data, dict) and any(
        key in data for key in ("boys_hostels", "girls_hostels", "common_facilities")
    ):
        common_facilities = data.get("common_facilities")
        if isinstance(common_facilities, dict):
            chunks.append(
                create_knowledge_chunk(
                    counter=counter,
                    source=source,
                    domain="hostel",
                    section=section,
                    title="Common Hostel Facilities",
                    record=common_facilities,
                )
            )
            counter += 1

        for hostel_group in ("boys_hostels", "girls_hostels"):
            hostels = data.get(hostel_group)
            if not isinstance(hostels, dict):
                continue
            for hostel_name, hostel_data in hostels.items():
                if not isinstance(hostel_data, dict):
                    continue
                chunks.append(
                    create_knowledge_chunk(
                        counter=counter,
                        source=source,
                        domain="hostel",
                        section=f"{section} / {normalize_label(hostel_group)}",
                        title=hostel_name,
                        record=hostel_data,
                    )
                )
                counter += 1
        return chunks

    if isinstance(data, dict) and all(
        isinstance(value, dict) for value in data.values()
    ):
        for item_name, item_data in data.items():
            chunks.append(
                create_knowledge_chunk(
                    counter=counter,
                    source=source,
                    domain=domain,
                    section=section,
                    title=item_name,
                    record=item_data,
                )
            )
            counter += 1
        return chunks

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            title = item.get("name") or item.get("title") or f"Item {counter}"
            chunks.append(
                create_knowledge_chunk(
                    counter=counter,
                    source=source,
                    domain=domain,
                    section=section,
                    title=title,
                    record=item,
                )
            )
            counter += 1
        return chunks

    if isinstance(data, dict):
        chunks.append(
            create_knowledge_chunk(
                counter=counter,
                source=source,
                domain=domain,
                section=section,
                title=section,
                record=data,
            )
        )

    return chunks


@lru_cache(maxsize=1)
def load_local_knowledge_chunks(path: str = "llm_main_prompt.txt") -> tuple[KnowledgeChunk, ...]:
    raw_text = load_local_knowledge_text(path)
    if not raw_text:
        return ()

    chunks: list[KnowledgeChunk] = []
    counter = 1

    for section_label, data in extract_json_sections(raw_text):
        section_chunks = chunk_json_section(
            data,
            section_label=section_label,
            source=path,
            start_counter=counter,
        )
        chunks.extend(section_chunks)
        counter += len(section_chunks)

    return tuple(chunks)


def should_search_web(prompt: str) -> bool:
    lower_prompt = prompt.lower()
    return is_tiet_related_prompt(prompt) and any(
        pattern in lower_prompt for pattern in TIME_SENSITIVE_PATTERNS
    )


def is_tiet_related_prompt(prompt: str) -> bool:
    lower_prompt = prompt.lower()
    return any(term in lower_prompt for term in TIET_CONTEXT_TERMS)


def build_search_query(prompt: str) -> str:
    prompt = normalize_whitespace(prompt)
    return f"{prompt} Thapar Institute of Engineering and Technology TIET Patiala"


def fetch_url(url: str, timeout: float, max_bytes: int = 180_000) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        raw = response.read(max_bytes)
    return raw.decode(charset, errors="replace")


def extract_page_text(markup: str, max_chars: int = 1200) -> str:
    parser = HTMLTextParser(max_chars=max_chars)
    parser.feed(markup)
    parser.close()
    return parser.get_text()


def search_web(
    query: str,
    *,
    max_results: int = 4,
    timeout: float = 6,
    fetch_pages: int = 2,
) -> list[SearchResult]:
    candidate_queries = [
        f"site:thapar.edu {query}",
        query,
    ]
    candidates: dict[str, SearchResult] = {}

    for candidate_query in candidate_queries:
        for result in fetch_search_results(candidate_query, timeout=timeout):
            if result.url in candidates:
                continue
            if not is_allowed_tiet_source(result):
                continue
            candidates[result.url] = result
            if len(candidates) >= max_results * 3:
                break
        if len(candidates) >= max_results:
            break

    ranked_results = sorted(
        candidates.values(),
        key=lambda result: score_tiet_source(result, query),
        reverse=True,
    )[:max_results]

    enriched_results: list[SearchResult] = []

    for index, result in enumerate(ranked_results):
        page_text = ""
        if index < fetch_pages:
            try:
                page_text = extract_page_text(fetch_url(result.url, timeout=timeout / 2))
            except (urllib.error.URLError, TimeoutError, OSError, UnicodeDecodeError):
                page_text = ""

        enriched_results.append(
            SearchResult(
                title=result.title,
                url=result.url,
                snippet=result.snippet,
                text=page_text,
            )
        )

    return enriched_results


def fetch_search_results(query: str, *, timeout: float = 6) -> list[SearchResult]:
    search_url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})

    try:
        markup = fetch_url(search_url, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, OSError):
        return []

    parser = DuckDuckGoResultParser()
    parser.feed(markup)
    parser.close()
    return parser.results


def get_hostname(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")


def hostname_matches(hostname: str, domains: Iterable[str]) -> bool:
    normalized_host = hostname.removeprefix("www.")
    return any(
        normalized_host == domain.removeprefix("www.")
        or normalized_host.endswith("." + domain.removeprefix("www."))
        for domain in domains
    )


def is_allowed_tiet_source(result: SearchResult) -> bool:
    hostname = get_hostname(result.url)
    if hostname_matches(hostname, BLOCKED_SOURCE_DOMAINS):
        return False
    return score_tiet_source(result, "") > 0


def score_tiet_source(result: SearchResult, query: str) -> int:
    hostname = get_hostname(result.url)
    haystack = " ".join([result.title, result.url, result.snippet, result.text]).lower()
    score = 0

    if hostname_matches(hostname, OFFICIAL_THAPAR_DOMAINS):
        score += 120

    if "thapar" in haystack:
        score += 50
    if "tiet" in haystack:
        score += 35
    if "patiala" in haystack:
        score += 10

    query_tokens = tokenize(query)
    haystack_tokens = set(tokenize(haystack))
    score += 3 * len(query_tokens & haystack_tokens)

    if result.url.lower().endswith(".pdf"):
        score += 8

    return score


def format_web_context(results: Iterable[SearchResult], query: str) -> str:
    results = list(results)
    if not results:
        return ""

    sections = [f"Search query: {query}"]
    for index, result in enumerate(results, start=1):
        source_text = result.text or result.snippet
        sections.append(
            "\n".join(
                [
                    f"[{index}] {result.title}",
                    f"URL: {result.url}",
                    f"Evidence: {source_text[:1400]}",
                ]
            )
        )
    return "\n\n".join(sections)


def build_student_context(student_details: dict[str, str] | None) -> str:
    if not student_details:
        return ""

    safe_details = {
        "name": student_details.get("name", "").strip(),
        "roll_no": student_details.get("roll_no", "").strip(),
        "branch": student_details.get("branch", "").strip(),
    }
    populated = [f"{label}: {value}" for label, value in safe_details.items() if value]
    if not populated:
        return ""
    return "Student profile for personalization only:\n" + "\n".join(populated)


def tokenize(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", value.lower())
        if token not in STOPWORDS
    }


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


def chunk_document_text(
    source: str,
    text: str,
    *,
    max_chars: int = 1600,
    overlap: int = 180,
) -> list[DocumentChunk]:
    text = normalize_whitespace(text)
    if not text:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    chunk_id = 1

    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + max_chars // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(DocumentChunk(source=source, chunk_id=chunk_id, text=chunk))
            chunk_id += 1
        if end >= len(text):
            break
        start = max(end - overlap, 0)

    return chunks


def prompt_mentions_documents(prompt: str) -> bool:
    lower_prompt = prompt.lower()
    return any(term in lower_prompt for term in DOCUMENT_REFERENCE_TERMS)


def select_relevant_document_chunks(
    prompt: str,
    chunks: Iterable[DocumentChunk],
    *,
    max_chunks: int = 5,
) -> list[DocumentChunk]:
    chunks = list(chunks)
    if not chunks:
        return []

    prompt_tokens = tokenize(prompt)
    scored_chunks: list[tuple[int, DocumentChunk]] = []

    for chunk in chunks:
        chunk_tokens = tokenize(chunk.text + " " + chunk.source)
        score = len(prompt_tokens & chunk_tokens)
        if score:
            scored_chunks.append((score, chunk))

    if scored_chunks:
        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:max_chunks]]

    if prompt_mentions_documents(prompt):
        return chunks[: min(max_chunks, len(chunks))]

    return []


def format_document_context(chunks: Iterable[DocumentChunk]) -> str:
    chunks = list(chunks)
    if not chunks:
        return ""

    sections = []
    for index, chunk in enumerate(chunks, start=1):
        sections.append(
            "\n".join(
                [
                    f"[D{index}] {chunk.source}, chunk {chunk.chunk_id}",
                    f"Evidence: {chunk.text[:1400]}",
                ]
            )
        )

    return "\n\n".join(sections)


def infer_query_domains(prompt: str) -> set[str]:
    prompt_tokens = tokenize(prompt)
    domains = set()

    if prompt_tokens & HOSTEL_QUERY_TERMS:
        domains.add("hostel")
    if prompt_tokens & FACULTY_QUERY_TERMS:
        domains.add("faculty")
    if prompt_tokens & CLUB_QUERY_TERMS:
        domains.add("club")
    if prompt_tokens & STAFF_QUERY_TERMS:
        domains.add("staff")

    return domains


def score_knowledge_chunk(prompt: str, chunk: KnowledgeChunk) -> int:
    prompt_tokens = tokenize(prompt)
    if not prompt_tokens:
        return 0

    title_tokens = tokenize(chunk.title)
    section_tokens = tokenize(chunk.section)
    body_tokens = tokenize(chunk.text)
    prompt_lower = prompt.lower()
    chunk_blob = " ".join([chunk.title, chunk.section, chunk.text]).lower()

    score = 0
    score += 6 * len(prompt_tokens & title_tokens)
    score += 3 * len(prompt_tokens & section_tokens)
    score += 2 * len(prompt_tokens & body_tokens)

    query_domains = infer_query_domains(prompt)
    if query_domains and chunk.domain in query_domains:
        score += 6

    if chunk.title.lower() in prompt_lower or prompt_lower in chunk_blob:
        score += 8

    return score


def estimate_token_count(value: str) -> int:
    return max(1, len(value) // 4)


def deduplicate_knowledge_chunks(
    chunks: Iterable[KnowledgeChunk],
) -> list[KnowledgeChunk]:
    unique_chunks: list[KnowledgeChunk] = []
    seen = set()

    for chunk in chunks:
        lines = [
            line
            for line in chunk.text.splitlines()
            if not line.startswith("Section: ")
        ]
        fingerprint = normalize_whitespace(
            f"{chunk.domain} {chunk.title} {' '.join(lines)}"
        ).lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique_chunks.append(chunk)

    return unique_chunks


def select_relevant_knowledge_chunks(
    prompt: str,
    chunks: Iterable[KnowledgeChunk],
    *,
    min_chunks: int = 3,
    max_chunks: int = 5,
    max_context_tokens: int = 1000,
) -> list[KnowledgeChunk]:
    chunks = list(chunks)
    if not chunks:
        return []

    prompt_tokens = tokenize(prompt)
    if not prompt_tokens:
        return []

    scored_chunks = sorted(
        (
            (score_knowledge_chunk(prompt, chunk), chunk)
            for chunk in deduplicate_knowledge_chunks(chunks)
        ),
        key=lambda item: item[0],
        reverse=True,
    )

    positive_matches = [chunk for score, chunk in scored_chunks if score > 0]
    if not positive_matches:
        return []

    selected: list[KnowledgeChunk] = []
    total_tokens = 0

    for chunk in positive_matches:
        chunk_tokens = estimate_token_count(chunk.text)
        if selected and total_tokens + chunk_tokens > max_context_tokens:
            continue
        selected.append(chunk)
        total_tokens += chunk_tokens
        if len(selected) >= max_chunks:
            break

    if len(selected) >= min_chunks:
        return selected

    preferred_domains = infer_query_domains(prompt)
    for _, chunk in scored_chunks:
        if chunk in selected:
            continue
        if preferred_domains and chunk.domain not in preferred_domains:
            continue
        chunk_tokens = estimate_token_count(chunk.text)
        if selected and total_tokens + chunk_tokens > max_context_tokens:
            continue
        selected.append(chunk)
        total_tokens += chunk_tokens
        if len(selected) >= min(max_chunks, min_chunks):
            break

    return selected


def format_knowledge_context(chunks: Iterable[KnowledgeChunk]) -> str:
    chunks = list(chunks)
    if not chunks:
        return ""

    sections = []
    for chunk in chunks:
        sections.append(
            "\n".join(
                [
                    f"[{chunk.chunk_id}] {chunk.title}",
                    f"Source: {chunk.source}",
                    f"Domain: {chunk.domain}",
                    f"Section: {chunk.section}",
                    chunk.text,
                ]
            )
        )

    return "\n\n".join(sections)


def build_system_message() -> str:
    today = datetime.now().strftime("%B %d, %Y")
    return f"{DEFAULT_RAG_SYSTEM_PROMPT} Current date: {today}."


def build_rag_user_message(
    *,
    user_prompt: str,
    knowledge_context: str = "",
    document_context: str = "",
    web_context: str = "",
    student_context: str = "",
) -> str:
    context_blocks = []

    if knowledge_context:
        context_blocks.append("LOCAL_KNOWLEDGE_CHUNKS:\n" + knowledge_context)
    if document_context:
        context_blocks.append("DOCUMENT_CONTEXT:\n" + document_context)
    if web_context:
        context_blocks.append("WEB_CONTEXT:\n" + web_context)
    if student_context:
        context_blocks.append(student_context)

    context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant context retrieved."

    return "\n".join(
        [
            "CONTEXT:",
            context_text,
            "",
            "QUESTION:",
            user_prompt,
            "",
            "INSTRUCTIONS:",
            "Answer clearly and concisely.",
            "Prefer structured responses when helpful.",
            "Do not hallucinate or use outside knowledge.",
            'If the answer is missing from the context, say "I don\'t have that information."',
            "If multiple chunks conflict, mention the uncertainty.",
            "Show the retrieved chunk ids and source labels used for the answer.",
        ]
    )


def build_api_messages(
    *,
    history: list[dict[str, str]],
    user_prompt: str,
    knowledge_context: str = "",
    student_details: dict[str, str] | None = None,
    web_context: str = "",
    document_context: str = "",
    max_history_messages: int = 8,
) -> list[dict[str, str]]:
    system_message = build_system_message()
    recent_history = [
        {"role": message["role"], "content": message["content"]}
        for message in history[-max_history_messages:]
        if message.get("role") in {"user", "assistant"} and message.get("content")
    ]
    rag_user_message = build_rag_user_message(
        user_prompt=user_prompt,
        knowledge_context=knowledge_context,
        document_context=document_context,
        web_context=web_context,
        student_context=build_student_context(student_details),
    )
    return [{"role": "system", "content": system_message}] + recent_history + [
        {"role": "user", "content": rag_user_message}
    ]
