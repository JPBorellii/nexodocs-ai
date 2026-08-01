"""Offline deterministic writers for canonical PDFs, CSVs and their catalog."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import IO, Literal, cast

from pypdf import PdfReader
from reportlab.lib.colors import HexColor  # pyright: ignore[reportMissingTypeStubs]
from reportlab.lib.pagesizes import A4  # pyright: ignore[reportMissingTypeStubs]
from reportlab.lib.styles import (  # pyright: ignore[reportMissingTypeStubs]
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import mm  # pyright: ignore[reportMissingTypeStubs]
from reportlab.pdfgen.canvas import Canvas  # pyright: ignore[reportMissingTypeStubs]
from reportlab.platypus import (  # pyright: ignore[reportMissingTypeStubs]
    BaseDocTemplate,
    Flowable,
    PageBreak,
    Paragraph,
    Spacer,
)

from .constants import CATALOG_VERSION, CONTROLLED_FILENAMES, GENERATOR_VERSION
from .models import CanonicalDocument, KnowledgeBaseError, load_document, required_string
from .validator import validate_canonical_documents, validate_generated_output


def repository_root() -> Path:
    """Return the repository root from this installed source-tree layout."""
    return Path(__file__).resolve().parents[3]


def load_canonical(root: Path) -> list[CanonicalDocument]:
    """Load canonical documents in deterministic path order."""
    canonical = root / "knowledge_base" / "canonical"
    paths = sorted((canonical / "policies").glob("*.json")) + sorted(
        (canonical / "tables").glob("*.json")
    )
    documents = [load_document(path, path.relative_to(root).as_posix()) for path in paths]
    validate_canonical_documents(documents, canonical / "canonical.schema.json")
    return sorted(documents, key=lambda document: document.document_id)


def _write_if_changed(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == content:
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _pdf_canvas(
    filename: str | IO[bytes],
    pagesize: tuple[float, float] | None = None,
    bottomup: Literal[0, 1] | bool = 1,
    **_: object,
) -> Canvas:
    """Adapt ReportLab's dynamic canvas factory to fixed deterministic inputs."""
    canvas = Canvas(
        filename, pagesize=pagesize or A4, bottomup=bottomup, invariant=1, pageCompression=0
    )
    canvas.setAuthor("Nexo Saúde Integrada - demonstração fictícia")
    canvas.setCreator("NexoDocs AI Knowledge Base Generator")
    canvas.setProducer("NexoDocs AI 1.0.0")
    return canvas


def _validate_cp1252(value: str) -> None:
    try:
        value.encode("cp1252")
    except UnicodeEncodeError as exc:
        raise KnowledgeBaseError("Texto contém caractere incompatível com Helvetica") from exc


def _render_pdf(document: CanonicalDocument, target: Path) -> None:
    data = document.data
    for value in _all_strings(data):
        _validate_cp1252(value)
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "NexoTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=16, leading=20
    )
    normal = ParagraphStyle(
        "NexoBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.3, leading=13
    )
    heading = ParagraphStyle(
        "NexoHeading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14
    )
    notice = ParagraphStyle(
        "NexoNotice",
        parent=normal,
        fontName="Helvetica-Bold",
        backColor=HexColor("#E8F1FA"),
        borderColor=HexColor("#4B78A8"),
        borderWidth=0.5,
        borderPadding=6,
    )
    metadata = ParagraphStyle("NexoMeta", parent=normal, fontSize=8, leading=10)
    story: list[Flowable] = [
        Paragraph(required_string(data, "title"), title),
        Spacer(1, 3 * mm),
        Paragraph(
            f"ID: {document.document_id}<br/>Versão: {required_string(data, 'version')}<br/>"
            f"Vigência fictícia: {required_string(data, 'effective_date')}<br/>"
            f"Responsável: {required_string(data, 'owner_area')}<br/>"
            f"Contato: {required_string(data, 'owner_contact')}",
            metadata,
        ),
        Spacer(1, 3 * mm),
        Paragraph(required_string(data, "fictitious_notice"), notice),
        Spacer(1, 4 * mm),
    ]
    content = cast(dict[str, object], data["content"])
    sections = cast(list[dict[str, object]], content["sections"])
    for index, section in enumerate(sections):
        story.append(Paragraph(cast(str, section["heading"]), heading))
        for paragraph in cast(list[str], section["paragraphs"]):
            story.extend([Paragraph(paragraph, normal), Spacer(1, 2.2 * mm)])
        if index == 2:
            story.append(PageBreak())

    def footer(canvas: Canvas, document_template: object) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(
            18 * mm, 12 * mm, f"{document.document_id} | versão {required_string(data, 'version')}"
        )
        canvas.drawRightString(192 * mm, 12 * mm, f"Página {canvas.getPageNumber()}")
        canvas.restoreState()

    template = BaseDocTemplate(
        str(target),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
    )
    from reportlab.platypus import Frame, PageTemplate  # pyright: ignore[reportMissingTypeStubs]

    frame = Frame(
        template.leftMargin, template.bottomMargin, template.width, template.height, id="body"
    )
    template.addPageTemplates(PageTemplate(id="Nexo", frames=[frame], onPage=footer))
    template.build(story, canvasmaker=_pdf_canvas)  # pyright: ignore[reportArgumentType] - ReportLab's canvas factory callback uses dynamically typed keyword arguments.


def _all_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in cast(list[object], value) for item in _all_strings(child)]
    if isinstance(value, Mapping):
        return [
            item
            for child in cast(Mapping[str, object], value).values()
            for item in _all_strings(child)
        ]
    return []


def _render_csv(document: CanonicalDocument) -> bytes:
    content = cast(dict[str, object], document.data["content"])
    columns = cast(list[str], content["columns"])
    rows = cast(list[dict[str, object]], content["rows"])
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=columns, delimiter=";", lineterminator="\n", quoting=csv.QUOTE_MINIMAL
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _catalog(documents: list[CanonicalDocument], output_root: Path) -> bytes:
    entries: list[dict[str, object]] = []
    for document in documents:
        path = output_root / "source" / document.filename
        entry = {
            key: document.data[key]
            for key in (
                "document_id",
                "title",
                "description",
                "category",
                "format",
                "version",
                "effective_date",
                "owner_area",
                "owner_contact",
                "language",
                "classification",
                "fictitious",
                "source_type",
                "locator_strategy",
            )
        }
        entry.update(
            {
                "filename": document.filename,
                "canonical_source": document.source,
                "sha256": _sha256(path),
                "byte_size": path.stat().st_size,
            }
        )
        if document.format == "pdf":
            entry["page_count"] = len(PdfReader(path).pages)
        else:
            entry["row_count"] = len(
                cast(list[object], cast(dict[str, object], document.data["content"])["rows"])
            )
        entries.append(entry)
    catalog = {
        "schema_version": "1.0",
        "catalog_version": CATALOG_VERSION,
        "generator_version": GENERATOR_VERSION,
        "documents": entries,
    }
    return (json.dumps(catalog, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _generate_to(root: Path, output_root: Path) -> None:
    documents = load_canonical(root)
    source = output_root / "source"
    source.mkdir(parents=True, exist_ok=True)
    for document in documents:
        path = source / document.filename
        if document.format == "pdf":
            _render_pdf(document, path)
        else:
            _write_if_changed(path, _render_csv(document))
    _write_if_changed(output_root / "metadata" / "catalog.json", _catalog(documents, output_root))
    validate_generated_output(root, output_root, documents)


def generate(root: Path | None = None, output_dir: Path | None = None) -> None:
    """Generate a validated knowledge base, writing only controlled outputs."""
    project_root = root or repository_root()
    destination = output_dir or project_root / "knowledge_base"
    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary) / "knowledge_base"
        _generate_to(project_root, staged)
        for filename in CONTROLLED_FILENAMES:
            _write_if_changed(
                destination / "source" / filename, (staged / "source" / filename).read_bytes()
            )
        _write_if_changed(
            destination / "metadata" / "catalog.json",
            (staged / "metadata" / "catalog.json").read_bytes(),
        )


def check(root: Path | None = None, output_dir: Path | None = None) -> None:
    """Regenerate into a temporary directory and compare controlled artifacts byte-for-byte."""
    project_root = root or repository_root()
    destination = output_dir or project_root / "knowledge_base"
    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary) / "knowledge_base"
        _generate_to(project_root, staged)
        expected = set(CONTROLLED_FILENAMES)
        actual = {path.name for path in (destination / "source").glob("*") if path.is_file()}
        if actual != expected:
            raise KnowledgeBaseError(
                f"Artefatos de origem divergentes: esperado {sorted(expected)}, encontrado {sorted(actual)}"
            )
        for filename in expected:
            if (destination / "source" / filename).read_bytes() != (
                staged / "source" / filename
            ).read_bytes():
                raise KnowledgeBaseError(f"Artefato fora de sincronização: {filename}")
        if (destination / "metadata" / "catalog.json").read_bytes() != (
            staged / "metadata" / "catalog.json"
        ).read_bytes():
            raise KnowledgeBaseError("Catálogo fora de sincronização")
