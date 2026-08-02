"""Closed prompt loading and deterministic data-only rendering."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .constants import ANSWER_PLACEHOLDERS, PROMPT_VERSION, RENDERING_STRATEGY_VERSION
from .context_builder import serialize_evidence
from .models import EvidenceBlock, RagError

_PLACEHOLDER = re.compile(r"\{\{[A-Z_]+\}\}")


def prompt_directory() -> Path:
    return Path(__file__).resolve().parents[3] / "prompts"


def load_prompts(root: Path | None = None) -> tuple[str, str]:
    directory = prompt_directory() if root is None else root / "prompts"
    system = (directory / "rag_system_v1.txt").read_text(encoding="utf-8")
    answer = (directory / "rag_answer_v1.txt").read_text(encoding="utf-8")
    found: set[str] = set(_PLACEHOLDER.findall(answer))
    if (
        found != set(ANSWER_PLACEHOLDERS)
        or any(answer.count(value) != 1 for value in ANSWER_PLACEHOLDERS)
        or _PLACEHOLDER.search(system)
    ):
        raise RagError("Placeholders de prompt inválidos")
    return system, answer


def prompt_sha256(system: str, answer: str) -> str:
    value = b"\x1f".join(
        (
            PROMPT_VERSION.encode(),
            system.encode(),
            answer.encode(),
            RENDERING_STRATEGY_VERSION.encode(),
        )
    )
    return hashlib.sha256(value).hexdigest()


def render_answer_prompt(
    template: str,
    query: str,
    evidence: tuple[EvidenceBlock, ...],
    schema: dict[str, object],
    maximum: int,
) -> str:
    replacements = {
        "{{QUERY_JSON}}": json.dumps(query, ensure_ascii=False),
        "{{EVIDENCE_JSON}}": serialize_evidence(evidence),
        "{{OUTPUT_SCHEMA_JSON}}": json.dumps(
            schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "{{MAX_ANSWER_CHARACTERS}}": str(maximum),
    }
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    if _PLACEHOLDER.search(rendered):
        raise RagError("Placeholder permaneceu após renderização")
    return rendered
