"""Validate checked-in processed knowledge-base artifacts without writing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nexodocs_ai.knowledge_base.generator import repository_root
from nexodocs_ai.processing.models import ProcessingError
from nexodocs_ai.processing.validator import validate_processed


def main() -> int:
    """Return non-zero when processed artifacts are invalid."""
    try:
        validate_processed(repository_root())
    except ProcessingError as exc:
        print(f"Processed knowledge base validation failed: {exc}")
        return 1
    print("Processed knowledge base validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
