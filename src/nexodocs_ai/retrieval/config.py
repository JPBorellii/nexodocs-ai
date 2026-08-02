"""Explicit environment configuration without dotenv loading."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping

from .constants import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_PER_DOCUMENT,
    DEFAULT_MAX_TOP_K,
    DEFAULT_OPENAI_MAX_RETRIES,
    DEFAULT_TOP_K,
)
from .models import ConfigurationError, RetrievalConfig


def _integer(values: Mapping[str, str], name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(values.get(name, str(default)))
    except ValueError as exc:
        raise ConfigurationError(f"{name} deve ser inteiro") from exc
    if value < minimum:
        qualifier = "n\u00e3o negativo" if minimum == 0 else "positivo"
        raise ConfigurationError(f"{name} deve ser {qualifier}")
    return value


def _positive_float(values: Mapping[str, str], name: str, default: float) -> float:
    try:
        value = float(values.get(name, str(default)))
    except ValueError as exc:
        raise ConfigurationError(f"{name} deve ser numérico") from exc
    if not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{name} deve ser finito e positivo")
    return value


def load_config(values: Mapping[str, str] | None = None) -> RetrievalConfig:
    """Read and validate explicit environment values without reading dotenv files."""
    env = os.environ if values is None else values
    threshold = env.get("RETRIEVAL_SCORE_THRESHOLD", "").strip()
    try:
        parsed_threshold = None if not threshold else float(threshold)
    except ValueError as exc:
        raise ConfigurationError("RETRIEVAL_SCORE_THRESHOLD deve ser numérico") from exc
    config = RetrievalConfig(
        app_env=env.get("APP_ENV", "development"),
        embedding_provider=env.get("EMBEDDING_PROVIDER", "openai"),
        openai_api_key=env.get("OPENAI_API_KEY", ""),
        embedding_model=env.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        embedding_dimensions=_integer(
            env, "OPENAI_EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS
        ),
        openai_timeout_seconds=_positive_float(env, "OPENAI_TIMEOUT_SECONDS", 30.0),
        openai_max_retries=_integer(
            env, "OPENAI_MAX_RETRIES", DEFAULT_OPENAI_MAX_RETRIES, minimum=0
        ),
        embedding_batch_size=_integer(env, "EMBEDDING_BATCH_SIZE", 32),
        qdrant_mode=env.get("QDRANT_MODE", "local"),
        qdrant_url=env.get("QDRANT_URL", ""),
        qdrant_api_key=env.get("QDRANT_API_KEY", ""),
        qdrant_path=env.get("QDRANT_PATH", "data/qdrant"),
        collection_name=env.get("QDRANT_COLLECTION_NAME", DEFAULT_COLLECTION_NAME),
        qdrant_timeout_seconds=_integer(env, "QDRANT_TIMEOUT_SECONDS", 10),
        top_k=_integer(env, "RETRIEVAL_TOP_K", DEFAULT_TOP_K),
        max_top_k=_integer(env, "RETRIEVAL_MAX_TOP_K", DEFAULT_MAX_TOP_K),
        max_per_document=_integer(env, "RETRIEVAL_MAX_PER_DOCUMENT", DEFAULT_MAX_PER_DOCUMENT),
        score_threshold=parsed_threshold,
    )
    if config.embedding_provider not in {"openai", "fake"} or config.qdrant_mode not in {
        "memory",
        "local",
        "remote",
    }:
        raise ConfigurationError("Provedor ou modo Qdrant inválido")
    if parsed_threshold is not None and (
        not math.isfinite(parsed_threshold) or not -1.0 <= parsed_threshold <= 1.0
    ):
        raise ConfigurationError("RETRIEVAL_SCORE_THRESHOLD deve estar entre -1 e 1")
    if config.embedding_provider == "fake" and config.app_env != "test":
        raise ConfigurationError("Provedor fake é permitido somente em APP_ENV=test")
    if config.qdrant_mode == "memory" and config.app_env != "test":
        raise ConfigurationError("Qdrant memory é permitido somente em APP_ENV=test")
    if config.qdrant_mode == "remote" and not config.qdrant_url:
        raise ConfigurationError("QDRANT_URL é obrigatório no modo remote")
    if not config.collection_name.strip():
        raise ConfigurationError("QDRANT_COLLECTION_NAME não pode ser vazio")
    if config.top_k > config.max_top_k:
        raise ConfigurationError("RETRIEVAL_TOP_K não pode exceder RETRIEVAL_MAX_TOP_K")
    return config


def load_plan_config(values: Mapping[str, str] | None = None) -> RetrievalConfig:
    """Load only non-sensitive settings required by the offline OpenAI index plan."""
    env = os.environ if values is None else values
    safe_values = {
        "APP_ENV": env.get("APP_ENV", "development"),
        "EMBEDDING_PROVIDER": "openai",
        "OPENAI_API_KEY": "",
        "OPENAI_EMBEDDING_MODEL": env.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "OPENAI_EMBEDDING_DIMENSIONS": env.get(
            "OPENAI_EMBEDDING_DIMENSIONS", str(DEFAULT_EMBEDDING_DIMENSIONS)
        ),
        "QDRANT_MODE": "local",
        "QDRANT_PATH": env.get("QDRANT_PATH", "data/qdrant"),
        "QDRANT_COLLECTION_NAME": env.get("QDRANT_COLLECTION_NAME", DEFAULT_COLLECTION_NAME),
    }
    return load_config(safe_values)
