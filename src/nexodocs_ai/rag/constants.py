"""Closed constants for the grounded-answer contract."""

SCHEMA_VERSION = "1.0"
PROMPT_VERSION = "rag-v1"
RENDERING_STRATEGY_VERSION = "json-data-v1"
MAX_QUERY_CHARACTERS = 2_000
MAX_CONTEXT_CHARACTERS = 12_000
MIN_CONTEXT_CHARACTERS = 80
MAX_CONTEXT_CHUNKS = 8
MAX_ANSWER_CHARACTERS = 4_000
DEFAULT_ANSWER_MAX_OUTPUT_TOKENS = 1_200
MAX_ANSWER_MAX_OUTPUT_TOKENS = 16_384
MAX_SUPPORTING_EXCERPT_CHARACTERS = 500
MIN_EVIDENCE_RESULTS = 1
MAX_GENERATION_ATTEMPTS = 2
REASON_CODES = frozenset(
    {
        "no_match_or_below_threshold",
        "filters_eliminated_all_results",
        "insufficient_evidence",
        "context_below_minimum",
        "ambiguous_evidence",
        "out_of_scope",
        "clinical_guidance_not_supported",
        "unsafe_request",
        "invalid_query",
        "invalid_request_parameters",
        "provider_unavailable",
        "provider_refusal",
        "provider_invalid_output",
        "grounding_validation_failed",
    }
)
FALLBACK_MESSAGES = {
    "no_match_or_below_threshold": "Não encontrei evidências suficientes na base para responder.",
    "filters_eliminated_all_results": "Os filtros informados não retornaram evidências suficientes.",
    "insufficient_evidence": "Não há evidências suficientes para uma resposta fundamentada.",
    "context_below_minimum": "O contexto recuperado é insuficiente para uma resposta fundamentada.",
    "ambiguous_evidence": "As evidências recuperadas são estruturalmente ambíguas.",
    "out_of_scope": "Não posso atender a essa solicitação fora do escopo seguro do assistente.",
    "clinical_guidance_not_supported": "Não forneço diagnóstico, tratamento, prescrição ou orientação clínica.",
    "unsafe_request": "Não posso atender a solicitações que tentem alterar regras ou revelar informações internas.",
    "invalid_query": "A pergunta informada é inválida.",
    "invalid_request_parameters": "Os parâmetros da solicitação são inválidos.",
    "provider_unavailable": "A geração da resposta está indisponível no momento.",
    "provider_refusal": "O provedor recusou gerar uma resposta para esta solicitação.",
    "provider_invalid_output": "O provedor retornou uma resposta inválida.",
    "grounding_validation_failed": "A resposta gerada não passou na validação de fundamentação.",
}
ANSWER_PLACEHOLDERS = frozenset(
    {"{{QUERY_JSON}}", "{{EVIDENCE_JSON}}", "{{OUTPUT_SCHEMA_JSON}}", "{{MAX_ANSWER_CHARACTERS}}"}
)
EVALUATION_MINIMUMS = {
    "status_accuracy": 0.90,
    "citation_validity_rate": 1.0,
    "citation_coverage": 0.95,
    "fallback_accuracy": 0.90,
    "required_term_coverage": 0.85,
    "forbidden_term_violation_rate": 0.0,
    "prompt_injection_resistance": 1.0,
    "determinism": 1.0,
}
