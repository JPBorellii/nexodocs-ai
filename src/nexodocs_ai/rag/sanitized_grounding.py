"""Non-reversible diagnostic projection for rejected grounding quotes.

The normalizations in this module are diagnostic signals only.  They are never
used by the acceptance validator.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import cast

from .constants import MAX_SUPPORTING_EXCERPT_CHARACTERS
from .models import EvidenceBlock, GeneratedCitationReference

MAX_DIAGNOSTIC_CHARACTERS = 16_000
_ZERO_WIDTH = frozenset({"\u200b", "\u200c", "\u200d", "\ufeff"})
_TYPOGRAPHIC_QUOTES = frozenset(
    {"\u2018", "\u2019", "\u201a", "\u201b", "\u201c", "\u201d", "\u201e", "\u201f"}
)
_NON_ASCII_DASHES = frozenset(
    {
        "\u058a",
        "\u05be",
        "\u1400",
        "\u1806",
        "\u2010",
        "\u2011",
        "\u2012",
        "\u2013",
        "\u2014",
        "\u2015",
        "\u2e17",
        "\u2e1a",
        "\u2e3a",
        "\u2e3b",
        "\u2e40",
        "\u301c",
        "\u3030",
        "\u30a0",
        "\ufe31",
        "\ufe32",
        "\ufe58",
        "\ufe63",
        "\uff0d",
    }
)
_TYPOGRAPHIC_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2026": "...",
        **{dash: "-" for dash in _NON_ASCII_DASHES},
    }
)


class LengthBucket(StrEnum):
    EMPTY = "EMPTY"
    VERY_SHORT = "VERY_SHORT"
    SHORT = "SHORT"
    MEDIUM = "MEDIUM"
    LONG = "LONG"
    OVER_LIMIT = "OVER_LIMIT"


class OverlapBucket(StrEnum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXACT = "EXACT"


class SanitizedRootCauseClass(StrEnum):
    EXACT_MATCH_UNEXPECTED = "EXACT_MATCH_UNEXPECTED"
    CASE_ONLY_DIFFERENCE = "CASE_ONLY_DIFFERENCE"
    WHITESPACE_ONLY_DIFFERENCE = "WHITESPACE_ONLY_DIFFERENCE"
    UNICODE_NORMALIZATION_DIFFERENCE = "UNICODE_NORMALIZATION_DIFFERENCE"
    TYPOGRAPHIC_DIFFERENCE = "TYPOGRAPHIC_DIFFERENCE"
    PUNCTUATION_ONLY_DIFFERENCE = "PUNCTUATION_ONLY_DIFFERENCE"
    WRONG_CITATION_ASSOCIATION = "WRONG_CITATION_ASSOCIATION"
    NORMALIZED_MATCH_IN_OTHER_CITATION = "NORMALIZED_MATCH_IN_OTHER_CITATION"
    HIGH_OVERLAP_NON_LITERAL = "HIGH_OVERLAP_NON_LITERAL"
    MEDIUM_OVERLAP_NON_LITERAL = "MEDIUM_OVERLAP_NON_LITERAL"
    LOW_OVERLAP_NON_LITERAL = "LOW_OVERLAP_NON_LITERAL"
    NO_MEANINGFUL_OVERLAP = "NO_MEANINGFUL_OVERLAP"
    MULTIPLE_POSSIBLE_CAUSES = "MULTIPLE_POSSIBLE_CAUSES"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class SanitizedGroundingSignalSet:
    """Closed projection containing no source or generated text."""

    exact_match: bool
    casefold_match: bool
    whitespace_normalized_match: bool
    unicode_nfc_match: bool
    unicode_nfkc_match: bool
    punctuation_normalized_match: bool
    quote_and_dash_normalized_match: bool
    combined_normalized_match: bool
    same_citation_id: bool
    exists_in_other_citation_id: bool
    normalized_match_in_other_citation_id: bool
    quote_length_bucket: LengthBucket
    evidence_length_bucket: LengthBucket
    character_overlap_bucket: OverlapBucket
    token_overlap_bucket: OverlapBucket
    quote_contains_non_breaking_space: bool
    evidence_contains_non_breaking_space: bool
    quote_contains_zero_width_character: bool
    evidence_contains_zero_width_character: bool
    quote_contains_line_break: bool
    evidence_contains_line_break: bool
    quote_contains_typographic_quote: bool
    evidence_contains_typographic_quote: bool
    quote_contains_non_ascii_dash: bool
    evidence_contains_non_ascii_dash: bool
    normalization_input_truncated: bool
    validator_stage: str = "quote_membership"
    original_safe_error_code: str = "grounding_quote_not_in_evidence"

    def as_dict(self) -> dict[str, object]:
        """Return the closed JSON shape; enum values are plain strings."""
        return cast(dict[str, object], json.loads(json.dumps(asdict(self))))


def compact_whitespace(value: str) -> str:
    """Collapse Unicode whitespace and zero-width format separators to one space."""
    prepared = "".join(" " if character in _ZERO_WIDTH else character for character in value)
    return " ".join(prepared.split())


def normalize_typography(value: str) -> str:
    """Map curly quotes, apostrophes, ellipsis, and non-ASCII dashes to ASCII."""
    return value.translate(_TYPOGRAPHIC_TRANSLATION)


def normalize_punctuation(value: str) -> str:
    """Remove every Unicode punctuation code point without changing whitespace."""
    return "".join(
        character for character in value if not unicodedata.category(character).startswith("P")
    )


def combined_normalize(value: str) -> str:
    """Apply bounded NFKC, casefold, typography, punctuation, and whitespace rules."""
    return compact_whitespace(
        normalize_punctuation(normalize_typography(unicodedata.normalize("NFKC", value).casefold()))
    )


def length_bucket(value: str) -> LengthBucket:
    """Bucket length without retaining the exact character count."""
    size = len(value)
    if size == 0:
        return LengthBucket.EMPTY
    if size <= 25:
        return LengthBucket.VERY_SHORT
    if size <= 100:
        return LengthBucket.SHORT
    if size <= 250:
        return LengthBucket.MEDIUM
    if size <= MAX_SUPPORTING_EXCERPT_CHARACTERS:
        return LengthBucket.LONG
    return LengthBucket.OVER_LIMIT


def _bucket_ratio(ratio: float, *, exact: bool = False) -> OverlapBucket:
    if exact:
        return OverlapBucket.EXACT
    if ratio == 0:
        return OverlapBucket.NONE
    if ratio < 0.34:
        return OverlapBucket.LOW
    if ratio < 0.67:
        return OverlapBucket.MEDIUM
    return OverlapBucket.HIGH


def _character_overlap(quote: str, evidence: str) -> OverlapBucket:
    left = "".join(character for character in combined_normalize(quote) if not character.isspace())
    right = "".join(
        character for character in combined_normalize(evidence) if not character.isspace()
    )
    if not left or not right:
        return OverlapBucket.NONE
    if left == right:
        return OverlapBucket.EXACT
    left_counts, right_counts = Counter(left), Counter(right)
    shared = sum((left_counts & right_counts).values())
    return _bucket_ratio((2 * shared) / (len(left) + len(right)))


def _token_overlap(quote: str, evidence: str) -> OverlapBucket:
    left, right = set(combined_normalize(quote).split()), set(combined_normalize(evidence).split())
    if not left or not right:
        return OverlapBucket.NONE
    if left == right:
        return OverlapBucket.EXACT
    return _bucket_ratio(len(left & right) / len(left | right))


def _contains(needle: str, haystack: str, transform: Callable[[str], str]) -> bool:
    return bool(needle) and transform(needle) in transform(haystack)


def project_quote_failure(
    quote: str,
    citation_id: int,
    evidence: tuple[EvidenceBlock, ...],
) -> SanitizedGroundingSignalSet:
    """Project one quote and its available evidence into bounded safe signals."""
    by_id = {block.evidence_id: block for block in evidence}
    declared = by_id.get(citation_id)
    declared_text = "" if declared is None else declared.text
    truncated = (
        len(quote) > MAX_DIAGNOSTIC_CHARACTERS or len(declared_text) > MAX_DIAGNOSTIC_CHARACTERS
    )
    bounded_quote = quote[:MAX_DIAGNOSTIC_CHARACTERS]
    bounded_evidence = declared_text[:MAX_DIAGNOSTIC_CHARACTERS]
    others = tuple(
        block.text[:MAX_DIAGNOSTIC_CHARACTERS]
        for block in evidence
        if block.evidence_id != citation_id
    )

    exact = bool(quote) and declared is not None and quote in declared_text
    casefold = _contains(bounded_quote, bounded_evidence, str.casefold)
    whitespace = _contains(bounded_quote, bounded_evidence, compact_whitespace)
    nfc = _contains(
        bounded_quote, bounded_evidence, lambda value: unicodedata.normalize("NFC", value)
    )
    nfkc = _contains(
        bounded_quote, bounded_evidence, lambda value: unicodedata.normalize("NFKC", value)
    )
    punctuation = _contains(bounded_quote, bounded_evidence, normalize_punctuation)
    typography = _contains(bounded_quote, bounded_evidence, normalize_typography)
    combined = _contains(bounded_quote, bounded_evidence, combined_normalize)
    other_exact = bool(quote) and any(quote in value for value in others)
    other_normalized = bool(quote) and any(
        combined_normalize(bounded_quote) in combined_normalize(value) for value in others
    )
    return SanitizedGroundingSignalSet(
        exact,
        casefold,
        whitespace,
        nfc,
        nfkc,
        punctuation,
        typography,
        combined,
        declared is not None,
        other_exact,
        other_normalized,
        length_bucket(quote),
        length_bucket(declared_text),
        _character_overlap(bounded_quote, bounded_evidence),
        _token_overlap(bounded_quote, bounded_evidence),
        "\u00a0" in quote,
        "\u00a0" in declared_text,
        any(character in quote for character in _ZERO_WIDTH),
        any(character in declared_text for character in _ZERO_WIDTH),
        "\n" in quote or "\r" in quote,
        "\n" in declared_text or "\r" in declared_text,
        any(character in quote for character in _TYPOGRAPHIC_QUOTES),
        any(character in declared_text for character in _TYPOGRAPHIC_QUOTES),
        any(character in quote for character in _NON_ASCII_DASHES),
        any(character in declared_text for character in _NON_ASCII_DASHES),
        truncated or any(len(block.text) > MAX_DIAGNOSTIC_CHARACTERS for block in evidence),
    )


def classify_signal_set(signals: SanitizedGroundingSignalSet) -> SanitizedRootCauseClass:
    """Derive one deterministic diagnosis exclusively from sanitized signals."""
    if (
        signals.quote_length_bucket is LengthBucket.EMPTY
        or signals.evidence_length_bucket is LengthBucket.EMPTY
    ):
        return SanitizedRootCauseClass.INCONCLUSIVE
    if signals.exact_match:
        return SanitizedRootCauseClass.EXACT_MATCH_UNEXPECTED
    same_normalized = any(
        (
            signals.casefold_match,
            signals.whitespace_normalized_match,
            signals.unicode_nfc_match,
            signals.unicode_nfkc_match,
            signals.punctuation_normalized_match,
            signals.quote_and_dash_normalized_match,
            signals.combined_normalized_match,
        )
    )
    if signals.exists_in_other_citation_id:
        return SanitizedRootCauseClass.WRONG_CITATION_ASSOCIATION
    if signals.normalized_match_in_other_citation_id and not same_normalized:
        return SanitizedRootCauseClass.NORMALIZED_MATCH_IN_OTHER_CITATION
    whitespace_marker = any(
        (
            signals.quote_contains_non_breaking_space,
            signals.evidence_contains_non_breaking_space,
            signals.quote_contains_zero_width_character,
            signals.evidence_contains_zero_width_character,
            signals.quote_contains_line_break,
            signals.evidence_contains_line_break,
        )
    )
    if signals.whitespace_normalized_match and whitespace_marker and not signals.casefold_match:
        return SanitizedRootCauseClass.WHITESPACE_ONLY_DIFFERENCE
    if (
        signals.quote_and_dash_normalized_match
        and not signals.casefold_match
        and not signals.whitespace_normalized_match
    ):
        return SanitizedRootCauseClass.TYPOGRAPHIC_DIFFERENCE
    causes: list[SanitizedRootCauseClass] = []
    if signals.casefold_match:
        causes.append(SanitizedRootCauseClass.CASE_ONLY_DIFFERENCE)
    if signals.whitespace_normalized_match:
        causes.append(SanitizedRootCauseClass.WHITESPACE_ONLY_DIFFERENCE)
    if signals.unicode_nfc_match or signals.unicode_nfkc_match:
        causes.append(SanitizedRootCauseClass.UNICODE_NORMALIZATION_DIFFERENCE)
    if signals.quote_and_dash_normalized_match:
        causes.append(SanitizedRootCauseClass.TYPOGRAPHIC_DIFFERENCE)
    if signals.punctuation_normalized_match:
        causes.append(SanitizedRootCauseClass.PUNCTUATION_ONLY_DIFFERENCE)
    unique = list(dict.fromkeys(causes))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1 or signals.combined_normalized_match:
        return SanitizedRootCauseClass.MULTIPLE_POSSIBLE_CAUSES
    if signals.normalization_input_truncated:
        return SanitizedRootCauseClass.INCONCLUSIVE
    strongest = max(
        (signals.character_overlap_bucket, signals.token_overlap_bucket),
        key=lambda value: list(OverlapBucket).index(value),
    )
    if strongest in {OverlapBucket.HIGH, OverlapBucket.EXACT}:
        return SanitizedRootCauseClass.HIGH_OVERLAP_NON_LITERAL
    if strongest is OverlapBucket.MEDIUM:
        return SanitizedRootCauseClass.MEDIUM_OVERLAP_NON_LITERAL
    if strongest is OverlapBucket.LOW:
        return SanitizedRootCauseClass.LOW_OVERLAP_NON_LITERAL
    return SanitizedRootCauseClass.NO_MEANINGFUL_OVERLAP


def project_quote_failures(
    citations: tuple[GeneratedCitationReference, ...],
    evidence: tuple[EvidenceBlock, ...],
) -> tuple[SanitizedGroundingSignalSet, ...]:
    """Project every literal membership failure after the validator has rejected output."""
    by_id = {block.evidence_id: block for block in evidence}
    projected = [
        project_quote_failure(item.quote, item.citation_id, evidence)
        for item in citations
        if item.citation_id in by_id
        and bool(item.quote)
        and len(item.quote) <= MAX_SUPPORTING_EXCERPT_CHARACTERS
        and item.quote not in by_id[item.citation_id].text
    ]
    return tuple(sorted(projected, key=lambda item: json.dumps(item.as_dict(), sort_keys=True)))


def aggregate_classification(
    signal_sets: tuple[SanitizedGroundingSignalSet, ...],
) -> SanitizedRootCauseClass:
    """Combine per-quote diagnoses without retaining quote order or identity."""
    classifications = {classify_signal_set(item) for item in signal_sets}
    if not classifications:
        return SanitizedRootCauseClass.INCONCLUSIVE
    if len(classifications) == 1:
        return next(iter(classifications))
    return SanitizedRootCauseClass.MULTIPLE_POSSIBLE_CAUSES
