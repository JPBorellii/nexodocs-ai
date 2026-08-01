"""Typed processing records."""

from __future__ import annotations

from dataclasses import dataclass


class ProcessingError(ValueError):
    """Raised when deterministic processing invariants fail."""


@dataclass(frozen=True)
class PdfPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class CsvRow:
    row_number: int
    values: dict[str, str]


@dataclass(frozen=True)
class ProcessedOutput:
    chunks: list[dict[str, object]]
    manifest: dict[str, object]
