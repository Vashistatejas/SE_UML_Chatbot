from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_HF_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_HF_MODEL = "openai/gpt-oss-120b"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


@dataclass(frozen=True)
class ModelConfig:
    base_url: str | None
    api_key: str | None
    api_key_name: str | None
    model: str


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
