"""End-to-end tests for the deterministic fictitious knowledge-base pipeline."""

from __future__ import annotations

from pathlib import Path

from nexodocs_ai.knowledge_base.generator import check, generate, repository_root


def test_pipeline_generates_and_checks_temporary_output(tmp_path: Path) -> None:
    """The pipeline creates five deterministic artifacts and validates them."""
    root = repository_root()
    generate(root, tmp_path)
    assert len(list((tmp_path / "source").iterdir())) == 5
    first = {path.name: path.read_bytes() for path in (tmp_path / "source").iterdir()}
    generate(root, tmp_path)
    assert first == {path.name: path.read_bytes() for path in (tmp_path / "source").iterdir()}
    check(root, tmp_path)
