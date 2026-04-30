from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openai import OpenAI

from rag_data import RagChunk


DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path(".cache") / "tiet_rag"
DOMAIN_TERMS = {
    "hostel": {"hostel", "hostels", "hall", "mess", "warden", "accommodation", "room"},
    "faculty": {"faculty", "professor", "professors", "mentor", "research", "department"},
    "club": {"club", "clubs", "society", "societies", "chapter"},
    "staff": {"staff", "dean", "office", "admin"},
    "document": {"document", "pdf", "file", "upload", "uploaded"},
}


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    batch_size: int = 64


@dataclass(frozen=True)
class RetrievalHit:
    chunk: RagChunk
    score: float


class BaseEmbedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

    @property
    def signature(self) -> str:
        return f"{self.config.provider}:{self.config.model}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, config: EmbeddingConfig, *, api_key: str, base_url: str | None = None) -> None:
        super().__init__(config)
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        outputs: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            response = self.client.embeddings.create(model=self.config.model, input=batch)
            outputs.extend(item.embedding for item in response.data)
        return outputs


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install sentence-transformers to use local embedding retrieval."
            ) from error

        self.model = SentenceTransformer(config.model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


def get_embedder() -> BaseEmbedder:
    provider = os.environ.get("EMBEDDING_PROVIDER", "").strip().lower()
    if not provider:
        provider = "openai" if os.environ.get("OPENAI_API_KEY") else "sentence-transformer"

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY for embeddings."
            )
        model = os.environ.get("EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL)
        base_url = os.environ.get("OPENAI_EMBEDDING_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        return OpenAIEmbedder(
            EmbeddingConfig(provider="openai", model=model, batch_size=96),
            api_key=api_key,
            base_url=base_url,
        )

    if provider in {"sentence-transformer", "sentence-transformers", "huggingface"}:
        model = os.environ.get("EMBEDDING_MODEL", DEFAULT_SENTENCE_TRANSFORMER_MODEL)
        return SentenceTransformerEmbedder(
            EmbeddingConfig(provider="sentence-transformer", model=model, batch_size=64)
        )

    raise RuntimeError(f"Unsupported embedding provider: {provider}")


def ensure_numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Install numpy to build the FAISS vector store.") from error
    return np


def ensure_faiss():
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError("Install faiss-cpu to use semantic retrieval.") from error
    return faiss


def estimate_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def detect_domains(query: str) -> set[str]:
    query_lower = query.lower()
    return {
        domain
        for domain, terms in DOMAIN_TERMS.items()
        if any(term in query_lower for term in terms)
    }


def build_corpus_signature(chunks: Iterable[RagChunk], embedder: BaseEmbedder) -> str:
    digest = hashlib.sha256()
    digest.update(embedder.signature.encode("utf-8"))
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(chunk.domain.encode("utf-8"))
        digest.update(chunk.title.encode("utf-8"))
        digest.update(chunk.text.encode("utf-8"))
    return digest.hexdigest()


class FaissChunkStore:
    def __init__(self, *, index, chunks: list[RagChunk], embedder: BaseEmbedder, cache_dir: Path) -> None:
        self.index = index
        self.chunks = chunks
        self.embedder = embedder
        self.cache_dir = cache_dir

    @classmethod
    def load_or_create(
        cls,
        *,
        chunks: list[RagChunk],
        embedder: BaseEmbedder,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        namespace: str = "main_dataset",
    ) -> "FaissChunkStore":
        faiss = ensure_faiss()
        np = ensure_numpy()

        cache_dir.mkdir(parents=True, exist_ok=True)
        namespace_dir = cache_dir / namespace
        namespace_dir.mkdir(parents=True, exist_ok=True)

        index_path = namespace_dir / "index.faiss"
        chunks_path = namespace_dir / "chunks.json"
        manifest_path = namespace_dir / "manifest.json"
        signature = build_corpus_signature(chunks, embedder)

        if index_path.exists() and chunks_path.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("signature") == signature:
                index = faiss.read_index(str(index_path))
                cached_chunks = [
                    RagChunk.from_dict(payload)
                    for payload in json.loads(chunks_path.read_text(encoding="utf-8"))
                ]
                return cls(index=index, chunks=cached_chunks, embedder=embedder, cache_dir=namespace_dir)

        texts = [chunk.text for chunk in chunks]
        matrix = np.array(embedder.embed_texts(texts), dtype="float32")
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise RuntimeError("Embedding provider returned an invalid vector matrix.")

        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        faiss.write_index(index, str(index_path))
        chunks_path.write_text(
            json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "signature": signature,
                    "embedding_provider": embedder.config.provider,
                    "embedding_model": embedder.config.model,
                    "dimension": int(matrix.shape[1]),
                    "count": len(chunks),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return cls(index=index, chunks=chunks, embedder=embedder, cache_dir=namespace_dir)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_context_tokens: int = 1000,
        preferred_domains: set[str] | None = None,
    ) -> list[RetrievalHit]:
        faiss = ensure_faiss()
        np = ensure_numpy()

        if not self.chunks:
            return []

        preferred_domains = preferred_domains or set()
        query_vector = np.array([self.embedder.embed_query(query)], dtype="float32")
        faiss.normalize_L2(query_vector)

        candidate_count = min(len(self.chunks), max(top_k * 8, 24))
        scores, indices = self.index.search(query_vector, candidate_count)

        ordered_hits = [
            RetrievalHit(chunk=self.chunks[index], score=float(score))
            for score, index in zip(scores[0], indices[0], strict=False)
            if index >= 0
        ]

        filtered_hits = [
            hit for hit in ordered_hits if not preferred_domains or hit.chunk.domain in preferred_domains
        ]
        primary_hits = filtered_hits or ordered_hits

        selected: list[RetrievalHit] = []
        seen_fingerprints = set()
        total_tokens = 0

        for hit in primary_hits:
            fingerprint = (
                hit.chunk.domain,
                hit.chunk.title.strip().lower(),
                hit.chunk.text.strip().lower(),
            )
            if fingerprint in seen_fingerprints:
                continue

            chunk_tokens = estimate_token_count(hit.chunk.text)
            if selected and total_tokens + chunk_tokens > max_context_tokens:
                continue

            selected.append(hit)
            seen_fingerprints.add(fingerprint)
            total_tokens += chunk_tokens

            if len(selected) >= top_k:
                break

        return selected


def build_transient_faiss_store(
    *,
    chunks: list[RagChunk],
    embedder: BaseEmbedder,
) -> FaissChunkStore:
    faiss = ensure_faiss()
    np = ensure_numpy()

    texts = [chunk.text for chunk in chunks]
    matrix = np.array(embedder.embed_texts(texts), dtype="float32")
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise RuntimeError("Embedding provider returned an invalid vector matrix.")

    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return FaissChunkStore(index=index, chunks=chunks, embedder=embedder, cache_dir=Path("."))
