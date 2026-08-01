"""Tests for processing domain models."""

from nexodocs_ai.processing.models import CsvRow, PdfPage, ProcessingError


def test_processing_models_are_immutable() -> None:
    """Location records preserve their values."""
    assert PdfPage(1, "á").page_number == 1
    assert CsvRow(1, {"id": "x"}).values["id"] == "x"
    assert issubclass(ProcessingError, ValueError)
