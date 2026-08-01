"""Immutable rules for deterministic document processing."""

PIPELINE_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"
TARGET_CHARACTERS = 1000
MAX_CHARACTERS = 1200
OVERLAP_CHARACTERS = 150
CHUNKS_FILENAME = "chunks.jsonl"
MANIFEST_FILENAME = "manifest.json"
PDF_FOOTER_PREFIX = "Página "
CSV_LABELS = {
    "unit_id": "ID da unidade",
    "unit_name": "Unidade",
    "city": "Cidade",
    "plan_id": "ID do plano",
    "plan_name": "Plano",
    "coverage_type": "Tipo de cobertura",
    "status": "Status",
    "effective_date": "Vigência",
    "confirmation_required": "Confirmação necessária",
    "notes": "Observações",
    "area_id": "ID da área",
    "area_name": "Área",
    "responsibility": "Responsabilidade",
    "contact_email": "Contato",
    "availability": "Disponibilidade",
    "response_target": "Prazo de resposta",
}
