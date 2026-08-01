"""Tests for page-preserving PDF extraction."""

from nexodocs_ai.knowledge_base.generator import repository_root
from nexodocs_ai.processing.extractors import extract_pdf_pages


def test_extracts_every_pdf_page_with_text() -> None:
    """Generated policy PDFs expose their two textual pages."""
    path = repository_root() / "knowledge_base" / "source" / "politica_ferias_beneficios_v1_0.pdf"
    pages = extract_pdf_pages(path)
    assert [page.page_number for page in pages] == [1, 2]
    assert "Férias" in pages[0].text
