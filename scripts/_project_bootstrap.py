"""Bootstrap standalone scripts against the repository source tree."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_project() -> Path:
    """Expose ``<repository>/src`` once and return the repository root."""
    root = Path(__file__).resolve().parents[1]
    source = root / "src"
    if not source.is_dir():
        raise RuntimeError("Repository source directory was not found.")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return root
