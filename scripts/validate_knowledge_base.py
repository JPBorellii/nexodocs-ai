"""Validate the checked-in fictitious knowledge base without writing files."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nexodocs_ai.knowledge_base.generator import repository_root
from nexodocs_ai.knowledge_base.models import KnowledgeBaseError
from nexodocs_ai.knowledge_base.validator import validate_repository


def main() -> int:
    """Return a non-zero exit status for any validation failure."""
    try:
        validate_repository(repository_root())
    except KnowledgeBaseError as exc:
        print(f"Knowledge base validation failed: {exc}")
        return 1
    print("Knowledge base validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
