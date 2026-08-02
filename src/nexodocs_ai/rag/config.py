"""Explicit RAG configuration; this module never loads dotenv files."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from .constants import (
    MAX_ANSWER_CHARACTERS,
    MAX_CONTEXT_CHARACTERS,
    MAX_CONTEXT_CHUNKS,
    MIN_CONTEXT_CHARACTERS,
    MIN_EVIDENCE_RESULTS,
    PROMPT_VERSION,
)
from .models import RagError


def _integer(env: Mapping[str, str], name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(env.get(name, str(default)))
    except ValueError as exc:
        raise RagError(f"{name} deve ser inteiro") from exc
    if value < minimum:
        raise RagError(f"{name} fora do limite")
    return value


@dataclass(frozen=True)
class RagConfig:
    app_env: str
    answer_provider: str
    openai_answer_model: str = ""
    openai_api_key: str = field(default="", repr=False)
    openai_answer_timeout_seconds: float = 45.0
    openai_answer_max_retries: int = 2
    max_query_characters: int = 2_000
    max_context_characters: int = MAX_CONTEXT_CHARACTERS
    min_context_characters: int = MIN_CONTEXT_CHARACTERS
    max_context_chunks: int = MAX_CONTEXT_CHUNKS
    max_answer_characters: int = MAX_ANSWER_CHARACTERS
    max_supporting_excerpt_characters: int = 500
    min_evidence_results: int = MIN_EVIDENCE_RESULTS
    max_generation_attempts: int = 2
    prompt_version: str = PROMPT_VERSION
    include_debug_metadata: bool = False

    def safe_summary(self) -> dict[str, object]:
        return {
            "app_env": self.app_env,
            "answer_provider": self.answer_provider,
            "openai_answer_model": self.openai_answer_model,
            "prompt_version": self.prompt_version,
        }


def load_rag_config(values: Mapping[str, str] | None = None) -> RagConfig:
    env = os.environ if values is None else values
    try:
        timeout = float(env.get("OPENAI_ANSWER_TIMEOUT_SECONDS", "45"))
    except ValueError as exc:
        raise RagError("OPENAI_ANSWER_TIMEOUT_SECONDS deve ser numérico") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise RagError("OPENAI_ANSWER_TIMEOUT_SECONDS fora do limite")
    debug = env.get("RAG_INCLUDE_DEBUG_METADATA", "false").casefold()
    if debug not in {"true", "false"}:
        raise RagError("RAG_INCLUDE_DEBUG_METADATA deve ser booleano")
    config = RagConfig(
        app_env=env.get("APP_ENV", "development"),
        answer_provider=env.get("ANSWER_PROVIDER", "openai"),
        openai_answer_model=env.get("OPENAI_ANSWER_MODEL", ""),
        openai_api_key=env.get("OPENAI_API_KEY", ""),
        openai_answer_timeout_seconds=timeout,
        openai_answer_max_retries=_integer(env, "OPENAI_ANSWER_MAX_RETRIES", 2, 0),
        max_query_characters=_integer(env, "RAG_MAX_QUERY_CHARACTERS", 2_000),
        max_context_characters=_integer(env, "RAG_MAX_CONTEXT_CHARACTERS", MAX_CONTEXT_CHARACTERS),
        min_context_characters=_integer(env, "RAG_MIN_CONTEXT_CHARACTERS", MIN_CONTEXT_CHARACTERS),
        max_context_chunks=_integer(env, "RAG_MAX_CONTEXT_CHUNKS", MAX_CONTEXT_CHUNKS),
        max_answer_characters=_integer(env, "RAG_MAX_ANSWER_CHARACTERS", MAX_ANSWER_CHARACTERS),
        max_supporting_excerpt_characters=_integer(
            env, "RAG_MAX_SUPPORTING_EXCERPT_CHARACTERS", 500
        ),
        min_evidence_results=_integer(env, "RAG_MIN_EVIDENCE_RESULTS", MIN_EVIDENCE_RESULTS),
        max_generation_attempts=_integer(env, "RAG_MAX_GENERATION_ATTEMPTS", 2),
        prompt_version=env.get("RAG_PROMPT_VERSION", PROMPT_VERSION),
        include_debug_metadata=debug == "true",
    )
    if (
        config.answer_provider not in {"openai", "fake"}
        or config.max_context_characters > MAX_CONTEXT_CHARACTERS
        or config.min_context_characters > config.max_context_characters
        or config.max_generation_attempts > 2
        or config.prompt_version != PROMPT_VERSION
    ):
        raise RagError("Configuração RAG inválida")
    if config.answer_provider == "fake" and config.app_env != "test":
        raise RagError("Provedor fake é permitido somente em APP_ENV=test")
    return config
