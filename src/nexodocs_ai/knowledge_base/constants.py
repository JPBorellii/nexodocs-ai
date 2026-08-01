"""Shared immutable rules for the fictitious knowledge base."""

from __future__ import annotations

CANONICAL_SCHEMA_VERSION = "1.0"
CATALOG_VERSION = "1.0"
GENERATOR_VERSION = "1.0.0"
EFFECTIVE_DATE = "2026-01-01"
FICTITIOUS_NOTICE = (
    "Documento fictício criado exclusivamente para fins educacionais. Não representa políticas, "
    "dados ou operações de uma organização real."
)
DOCUMENT_IDS = frozenset(
    {"NSI-POL-OPS-001", "NSI-POL-PEO-001", "NSI-POL-PRV-001", "NSI-TAB-COV-001", "NSI-TAB-DIR-001"}
)
PDF_FILENAMES = frozenset(
    {
        "politica_cancelamentos_reagendamentos_v1_0.pdf",
        "politica_ferias_beneficios_v1_0.pdf",
        "politica_privacidade_protecao_dados_v1_0.pdf",
    }
)
CSV_HEADERS = {
    "NSI-TAB-COV-001": [
        "unit_id",
        "unit_name",
        "city",
        "plan_id",
        "plan_name",
        "coverage_type",
        "status",
        "effective_date",
        "confirmation_required",
        "notes",
    ],
    "NSI-TAB-DIR-001": [
        "area_id",
        "area_name",
        "responsibility",
        "contact_email",
        "availability",
        "response_target",
        "status",
    ],
}
ALLOWED_CATEGORIES = frozenset(
    {
        "operational-policy",
        "people-policy",
        "privacy-policy",
        "coverage-directory",
        "contact-directory",
    }
)
ALLOWED_FORMATS = frozenset({"pdf", "csv"})
FORBIDDEN_CELL_PREFIXES = ("=", "+", "-", "@")
CONTROLLED_FILENAMES = PDF_FILENAMES | frozenset(
    {"convenios_coberturas_por_unidade_v1_0.csv", "diretorio_areas_responsaveis_contatos_v1_0.csv"}
)
