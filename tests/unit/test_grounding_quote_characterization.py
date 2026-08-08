"""Synthetic characterization matrix for strict whitespace-only quote grounding."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from nexodocs_ai.rag.constants import MAX_SUPPORTING_EXCERPT_CHARACTERS
from nexodocs_ai.rag.grounding_diagnostics import GroundingValidationError
from nexodocs_ai.rag.models import (
    EvidenceBlock,
    GeneratedAnswer,
    GeneratedCitationReference,
)
from nexodocs_ai.rag.validator import validate_generated

QUOTE_MISMATCH = "grounding_quote_mismatch"
QUOTE_NOT_IN_EVIDENCE = "grounding_quote_not_in_evidence"


@dataclass(frozen=True)
class QuoteCase:
    name: str
    evidence: str
    quote: str
    expected_error: str | None


def _block(identifier: int, text: str) -> EvidenceBlock:
    return EvidenceBlock(
        identifier,
        f"synthetic-chunk-{identifier}",
        "SYNTHETIC-DOC",
        "Documento sintético",
        "synthetic.txt",
        "line:1",
        "Documento sintético — linha 1",
        1.0,
        text,
        "a" * 64,
    )


def _outcome(evidence: str, quote: str) -> str | None:
    generated = GeneratedAnswer(
        "Resposta sintética [1]",
        (GeneratedCitationReference(1, quote),),
    )
    try:
        validate_generated(generated, (_block(1, evidence),))
    except GroundingValidationError as exc:
        return exc.safe_error_code
    return None


CASES = (
    QuoteCase("literal_exatamente_igual", "prazo de cinco dias", "prazo de cinco dias", None),
    QuoteCase("substring_literal", "o prazo de cinco dias úteis", "prazo de cinco", None),
    QuoteCase(
        "diferenca_de_caixa", "Prazo de cinco dias", "prazo de cinco dias", QUOTE_NOT_IN_EVIDENCE
    ),
    QuoteCase("espacos_duplicados", "prazo de cinco dias", "prazo  de cinco dias", None),
    QuoteCase(
        "run_de_espacos_equivalente",
        "prazo administrativo",
        "prazo   administrativo",
        None,
    ),
    QuoteCase("tab_equivalente", "prazo administrativo", "prazo\tadministrativo", None),
    QuoteCase("lf_equivalente", "prazo administrativo", "prazo\nadministrativo", None),
    QuoteCase("crlf_equivalente", "prazo administrativo", "prazo\r\nadministrativo", None),
    QuoteCase(
        "espacos_removidos", "prazo de cinco dias", "prazodecincodias", QUOTE_NOT_IN_EVIDENCE
    ),
    QuoteCase("tabulacao_literal", "prazo\tde cinco dias", "prazo\tde cinco", None),
    QuoteCase("quebra_lf_literal", "primeira\nsegunda", "primeira\nsegunda", None),
    QuoteCase("quebra_crlf_literal", "primeira\r\nsegunda", "primeira\r\nsegunda", None),
    QuoteCase(
        "quebra_substituida_por_espaco",
        "primeira\nsegunda",
        "primeira segunda",
        None,
    ),
    QuoteCase("tab_substituida_por_espaco", "prazo\tcurto", "prazo curto", None),
    QuoteCase("crlf_substituido_por_espaco", "primeira\r\nsegunda", "primeira segunda", None),
    QuoteCase(
        "runs_mistos_de_whitespace",
        "regra\t \r\n  ficticia",
        "regra ficticia",
        None,
    ),
    QuoteCase("whitespace_inicial_preservado", "\tprazo", "  prazo", None),
    QuoteCase("whitespace_final_preservado", "prazo\n", "prazo  ", None),
    QuoteCase("espaco_nao_separavel", "prazo\u00a0curto", "prazo curto", None),
    QuoteCase("boundary_inicial_contra_prefixo", "falta", " alta ", QUOTE_NOT_IN_EVIDENCE),
    QuoteCase("boundary_final_contra_sufixo", "prazos", "prazo ", QUOTE_NOT_IN_EVIDENCE),
    QuoteCase("boundary_inicial_ausente", "alta", " alta ", QUOTE_NOT_IN_EVIDENCE),
    QuoteCase("boundary_final_ausente", "prazo", "prazo ", QUOTE_NOT_IN_EVIDENCE),
    QuoteCase("unicode_nfc_literal", "Café", "Café", None),
    QuoteCase("unicode_nfd_literal", "Cafe\u0301", "Cafe\u0301", None),
    QuoteCase("unicode_nfc_contra_nfd", "Café", "Cafe\u0301", QUOTE_NOT_IN_EVIDENCE),
    QuoteCase("unicode_nfkc_equivalente", "① item", "1 item", QUOTE_NOT_IN_EVIDENCE),
    QuoteCase("acento_combinado", "ação", "ac\u0327a\u0303o", QUOTE_NOT_IN_EVIDENCE),
    QuoteCase("aspas_retas_contra_curvas", 'plano "Nexo"', "plano “Nexo”", QUOTE_NOT_IN_EVIDENCE),
    QuoteCase("apostrofo_reto_contra_tipografico", "d'água", "d’água", QUOTE_NOT_IN_EVIDENCE),
    QuoteCase("hifen_ascii_literal", "prazo-limite", "prazo-limite", None),
    QuoteCase("en_dash_contra_hifen", "prazo–limite", "prazo-limite", QUOTE_NOT_IN_EVIDENCE),
    QuoteCase("em_dash_contra_hifen", "prazo—limite", "prazo-limite", QUOTE_NOT_IN_EVIDENCE),
    QuoteCase(
        "pontuacao_final_diferente", "prazo encerrado.", "prazo encerrado!", QUOTE_NOT_IN_EVIDENCE
    ),
    QuoteCase(
        "virgula_diferente", "sim, quando aplicável", "sim quando aplicável", QUOTE_NOT_IN_EVIDENCE
    ),
    QuoteCase(
        "ponto_e_virgula_diferente", "regra; exceção", "regra, exceção", QUOTE_NOT_IN_EVIDENCE
    ),
    QuoteCase(
        "elipse_ascii_contra_unicode",
        "aguarde... retorno",
        "aguarde… retorno",
        QUOTE_NOT_IN_EVIDENCE,
    ),
    QuoteCase(
        "elipse_unicode_contra_ascii",
        "aguarde… retorno",
        "aguarde... retorno",
        QUOTE_NOT_IN_EVIDENCE,
    ),
    QuoteCase(
        "truncado_no_inicio", "o prazo máximo é cinco dias", "prazo máximo é cinco dias", None
    ),
    QuoteCase("truncado_no_fim", "o prazo máximo é cinco dias", "o prazo máximo", None),
    QuoteCase(
        "palavras_removidas",
        "prazo máximo de cinco dias",
        "prazo cinco dias",
        QUOTE_NOT_IN_EVIDENCE,
    ),
    QuoteCase(
        "palavras_adicionadas",
        "prazo de cinco dias",
        "prazo máximo de cinco dias",
        QUOTE_NOT_IN_EVIDENCE,
    ),
    QuoteCase(
        "palavra_diferente",
        "prazo de cinco dias",
        "prazo de seis dias",
        QUOTE_NOT_IN_EVIDENCE,
    ),
    QuoteCase(
        "whitespace_e_conteudo_diferentes",
        "prazo\n de cinco dias",
        "prazo   de seis dias",
        QUOTE_NOT_IN_EVIDENCE,
    ),
    QuoteCase(
        "quote_inexistente",
        "prazo de cinco dias",
        "regra sem correspondencia",
        QUOTE_NOT_IN_EVIDENCE,
    ),
    QuoteCase(
        "parafrase_equivalente",
        "o prazo termina em cinco dias",
        "a duração é de cinco dias",
        QUOTE_NOT_IN_EVIDENCE,
    ),
    QuoteCase("quote_vazia", "texto sintético", "", QUOTE_MISMATCH),
    QuoteCase("quote_somente_espacos", "texto com  intervalo", "  ", None),
    QuoteCase(
        "quote_no_limite",
        "x" * MAX_SUPPORTING_EXCERPT_CHARACTERS,
        "x" * MAX_SUPPORTING_EXCERPT_CHARACTERS,
        None,
    ),
    QuoteCase(
        "quote_acima_do_limite",
        "x" * (MAX_SUPPORTING_EXCERPT_CHARACTERS + 1),
        "x" * (MAX_SUPPORTING_EXCERPT_CHARACTERS + 1),
        QUOTE_MISMATCH,
    ),
    QuoteCase("evidencia_vazia", "", "texto", QUOTE_NOT_IN_EVIDENCE),
    QuoteCase("unicode_visualmente_semelhante", "A regra", "А regra", QUOTE_NOT_IN_EVIDENCE),
    QuoteCase("zero_width_space", "pra\u200bzo", "prazo", QUOTE_NOT_IN_EVIDENCE),
    QuoteCase("carriage_return_literal", "primeira\rsegunda", "primeira\rsegunda", None),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_literal_quote_characterization_matrix(case: QuoteCase) -> None:
    assert _outcome(case.evidence, case.quote) == case.expected_error


def test_quote_is_checked_only_against_its_declared_citation_id() -> None:
    generated = GeneratedAnswer(
        "Resposta sintética [2]",
        (GeneratedCitationReference(2, "texto da primeira evidência"),),
    )
    with pytest.raises(GroundingValidationError) as failure:
        validate_generated(
            generated,
            (
                _block(1, "texto da primeira evidência"),
                _block(2, "texto da segunda evidência"),
            ),
        )
    assert failure.value.safe_error_code == QUOTE_NOT_IN_EVIDENCE


def test_quote_for_the_declared_citation_id_is_accepted() -> None:
    generated = GeneratedAnswer(
        "Resposta sintética [2]",
        (GeneratedCitationReference(2, "segunda evidência"),),
    )
    rendered, citations = validate_generated(
        generated,
        (
            _block(1, "texto da primeira evidência"),
            _block(2, "texto da segunda evidência"),
        ),
    )
    assert rendered == "Resposta sintética [1]"
    assert citations[0].supporting_excerpt == "segunda evidência"


def test_same_quote_is_accepted_for_two_distinct_citation_ids() -> None:
    generated = GeneratedAnswer(
        "Resposta sintética [1] [2]",
        (
            GeneratedCitationReference(1, "regra compartilhada"),
            GeneratedCitationReference(2, "regra compartilhada"),
        ),
    )
    _, citations = validate_generated(
        generated,
        (
            _block(1, "regra compartilhada no primeiro bloco"),
            _block(2, "regra compartilhada no segundo bloco"),
        ),
    )
    assert len(citations) == 2
