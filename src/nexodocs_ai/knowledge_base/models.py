"""Small typed parsing helpers for canonical knowledge-base documents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast


class KnowledgeBaseError(ValueError):
    """Raised when a knowledge-base invariant is violated."""


@dataclass(frozen=True)
class CanonicalDocument:
    """A parsed canonical document with its source-relative path."""

    data: dict[str, object]
    source: str

    @property
    def document_id(self) -> str:
        return required_string(self.data, "document_id")

    @property
    def filename(self) -> str:
        return required_string(self.data, "output_filename")

    @property
    def format(self) -> str:
        return required_string(self.data, "format")


def required_string(data: dict[str, object], key: str) -> str:
    """Return a non-empty string field or raise a domain-specific error."""
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeBaseError(f"Campo obrigatório inválido: {key}")
    return value


def load_document(path: Path, source: str) -> CanonicalDocument:
    """Load one UTF-8 JSON canonical document without accepting non-objects."""
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeBaseError(f"Não foi possível ler {source}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise KnowledgeBaseError(f"O documento canônico deve ser objeto: {source}")
    return CanonicalDocument(data=cast(dict[str, object], parsed), source=source)
