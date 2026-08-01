"""Tests for row-preserving CSV extraction."""

from nexodocs_ai.knowledge_base.generator import repository_root
from nexodocs_ai.processing.extractors import extract_csv_rows


def test_extracts_csv_rows_and_boolean_values() -> None:
    """Rows preserve source order and their textual boolean values."""
    path = (
        repository_root()
        / "knowledge_base"
        / "source"
        / "convenios_coberturas_por_unidade_v1_0.csv"
    )
    rows = extract_csv_rows(path, "NSI-TAB-COV-001", 12)
    assert rows[0].row_number == 1
    assert rows[0].values["confirmation_required"] == "True"
