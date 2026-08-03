"""Synthetic characterization of the non-reversible D04 projection."""

from __future__ import annotations

from nexodocs_ai.rag.constants import MAX_SUPPORTING_EXCERPT_CHARACTERS
from nexodocs_ai.rag.models import EvidenceBlock, GeneratedCitationReference
from nexodocs_ai.rag.sanitized_grounding import (
    LengthBucket,
    OverlapBucket,
    SanitizedRootCauseClass,
    aggregate_classification,
    classify_signal_set,
    combined_normalize,
    compact_whitespace,
    normalize_punctuation,
    normalize_typography,
    project_quote_failure,
    project_quote_failures,
)


def _block(identifier: int, text: str) -> EvidenceBlock:
    return EvidenceBlock(
        identifier,
        f"synthetic-chunk-{identifier}",
        f"synthetic-document-{identifier}",
        "Synthetic title",
        "synthetic.txt",
        "section=synthetic",
        "Synthetic",
        0.9,
        text,
        "a" * 64,
    )


def _class(quote: str, evidence: str) -> SanitizedRootCauseClass:
    signals = project_quote_failure(quote, 1, (_block(1, evidence),))
    return classify_signal_set(signals)


def test_diagnostic_normalizations_have_explicit_deterministic_contracts() -> None:
    assert compact_whitespace(" a\t b\r\n c\u00a0d\u200be ") == "a b c d e"
    assert normalize_typography("‘a’ “b” c–d—e…") == "'a' \"b\" c-d-e..."
    assert normalize_punctuation("sim, regra…") == "sim regra"
    assert combined_normalize("  “REGRA”\u00a0– item… ") == "regra item"


def test_case_whitespace_and_line_break_classifications() -> None:
    assert (
        _class("POLITICA INTERNA", "politica interna")
        is SanitizedRootCauseClass.CASE_ONLY_DIFFERENCE
    )
    for evidence in (
        "regra  interna",
        "regra\tinterna",
        "regra\ninterna",
        "regra\r\ninterna",
        "regra\u00a0interna",
        "regra\u200binterna",
    ):
        assert (
            _class("regra interna", evidence) is SanitizedRootCauseClass.WHITESPACE_ONLY_DIFFERENCE
        )


def test_unicode_nfc_nfd_and_nfkc_classifications() -> None:
    assert _class("café", "cafe\u0301") is SanitizedRootCauseClass.UNICODE_NORMALIZATION_DIFFERENCE
    assert _class("1 item", "① item") is SanitizedRootCauseClass.UNICODE_NORMALIZATION_DIFFERENCE


def test_typographic_quote_apostrophe_dash_and_ellipsis_classifications() -> None:
    for quote, evidence in (
        ('plano "Nexo"', "plano “Nexo”"),
        ("d'agua", "d’agua"),
        ("prazo-limite", "prazo–limite"),
        ("prazo-limite", "prazo—limite"),
        ("aguarde... retorno", "aguarde… retorno"),
    ):
        assert _class(quote, evidence) is SanitizedRootCauseClass.TYPOGRAPHIC_DIFFERENCE


def test_punctuation_only_final_and_comma_classifications() -> None:
    assert (
        _class("regra interna.", "regra interna!")
        is SanitizedRootCauseClass.PUNCTUATION_ONLY_DIFFERENCE
    )
    assert (
        _class("sim, quando aplicavel", "sim; quando aplicavel")
        is SanitizedRootCauseClass.PUNCTUATION_ONLY_DIFFERENCE
    )


def test_other_citation_exact_and_normalized_association() -> None:
    evidence = (_block(1, "conteudo distinto"), _block(2, "regra correta literal"))
    exact = project_quote_failure("regra correta", 1, evidence)
    assert exact.exists_in_other_citation_id is True
    assert classify_signal_set(exact) is SanitizedRootCauseClass.WRONG_CITATION_ASSOCIATION

    normalized = project_quote_failure(
        "REGRA CORRETA", 1, (_block(1, "conteudo distinto"), _block(2, "regra correta"))
    )
    assert normalized.exists_in_other_citation_id is False
    assert normalized.normalized_match_in_other_citation_id is True
    assert (
        classify_signal_set(normalized)
        is SanitizedRootCauseClass.NORMALIZED_MATCH_IN_OTHER_CITATION
    )


def test_overlap_classification_buckets_cover_high_medium_low_and_none() -> None:
    cases = (
        ("aaaa bbbb", "aaaa cccc bbbb", SanitizedRootCauseClass.HIGH_OVERLAP_NON_LITERAL),
        ("aaaa bbbb", "aaaa cccc", SanitizedRootCauseClass.MEDIUM_OVERLAP_NON_LITERAL),
        ("aaaa bbbb", "aaaa xxxxxxxxxxxxxxxxxxxx", SanitizedRootCauseClass.LOW_OVERLAP_NON_LITERAL),
        ("aaaa bbbb", "xxxx yyyy", SanitizedRootCauseClass.NO_MEANINGFUL_OVERLAP),
    )
    for quote, evidence, expected in cases:
        assert _class(quote, evidence) is expected


def test_multiple_causes_empty_over_limit_and_empty_evidence() -> None:
    assert (
        _class("REGRA INTERNA", "regra  interna")
        is SanitizedRootCauseClass.MULTIPLE_POSSIBLE_CAUSES
    )
    empty = project_quote_failure("", 1, (_block(1, "synthetic"),))
    assert empty.quote_length_bucket is LengthBucket.EMPTY
    assert classify_signal_set(empty) is SanitizedRootCauseClass.INCONCLUSIVE
    over = project_quote_failure(
        "x" * (MAX_SUPPORTING_EXCERPT_CHARACTERS + 1), 1, (_block(1, "synthetic"),)
    )
    assert over.quote_length_bucket is LengthBucket.OVER_LIMIT
    assert (
        project_quote_failure("synthetic", 1, (_block(1, ""),)).evidence_length_bucket
        is LengthBucket.EMPTY
    )


def test_multiple_invalid_quotes_and_evidence_are_projected_without_identity() -> None:
    citations = (
        GeneratedCitationReference(1, "aaaa bbbb"),
        GeneratedCitationReference(2, "REGRA INTERNA"),
    )
    evidence = (_block(1, "aaaa cccc"), _block(2, "regra interna"), _block(3, "unused"))
    projected = project_quote_failures(citations, evidence)
    assert len(projected) == 2
    assert aggregate_classification(projected) is SanitizedRootCauseClass.MULTIPLE_POSSIBLE_CAUSES
    assert all(item.validator_stage == "quote_membership" for item in projected)
    assert all(
        item.original_safe_error_code == "grounding_quote_not_in_evidence" for item in projected
    )


def test_signal_shape_contains_only_booleans_and_closed_buckets() -> None:
    data = project_quote_failure("regra interna", 1, (_block(1, "regra\u00a0interna"),)).as_dict()
    assert data["quote_length_bucket"] == "VERY_SHORT"
    assert data["character_overlap_bucket"] in {item.value for item in OverlapBucket}
    assert data["evidence_contains_non_breaking_space"] is True
    assert not any(value == "regra interna" for value in data.values())
