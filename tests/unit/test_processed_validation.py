"""Tests for processed artifact validation."""

from pathlib import Path

import pytest

from nexodocs_ai.knowledge_base.generator import repository_root
from nexodocs_ai.processing.models import ProcessingError
from nexodocs_ai.processing.pipeline import generate
from nexodocs_ai.processing.validator import validate_processed


def test_validator_rejects_an_extra_file(tmp_path: Path) -> None:
    """Processed output is a closed, controlled artifact set."""
    generate(repository_root(), tmp_path)
    (tmp_path / "extra.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ProcessingError, match="divergentes"):
        validate_processed(repository_root(), tmp_path)
