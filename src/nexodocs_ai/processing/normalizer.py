"""Conservative, human-readable text normalization."""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Normalize whitespace without changing semantic textual content."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(re.sub(r"[ \t]+", " ", line).rstrip() for line in normalized.split("\n"))
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
