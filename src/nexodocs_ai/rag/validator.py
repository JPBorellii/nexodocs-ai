"""Structural, quote and conservative lexical grounding validation."""

from __future__ import annotations

import re

from .constants import MAX_ANSWER_CHARACTERS, MAX_SUPPORTING_EXCERPT_CHARACTERS
from .grounding_diagnostics import GroundingErrorCode, GroundingValidationError
from .models import Citation, EvidenceBlock, GeneratedAnswer

_MARKER = re.compile(r"\[(\d+)\]")
_FORBIDDEN = re.compile(
    r"https?://|\b(?:api[_ -]?key|sk-[A-Za-z0-9]|ignore (?:as )?regras|diagnóstico|prescrição|dose)\b",
    re.I,
)


def validate_generated(
    answer: GeneratedAnswer,
    evidence: tuple[EvidenceBlock, ...],
    maximum: int = MAX_ANSWER_CHARACTERS,
) -> tuple[str, tuple[Citation, ...]]:
    if (
        not answer.answer.strip()
        or len(answer.answer) > maximum
        or "{{" in answer.answer
        or _FORBIDDEN.search(answer.answer)
    ):
        if not answer.answer.strip():
            code = GroundingErrorCode.SCHEMA_INVALID
        elif len(answer.answer) > maximum:
            code = GroundingErrorCode.ANSWER_TOO_LONG
        elif "{{" in answer.answer:
            code = GroundingErrorCode.SCHEMA_INVALID
        else:
            code = GroundingErrorCode.UNSUPPORTED_CLAIM
        raise GroundingValidationError(code)
    available = {block.evidence_id: block for block in evidence}
    declared: dict[int, str] = {}
    for item in answer.citations:
        if (
            item.citation_id in declared
            or item.citation_id not in available
            or not item.quote
            or len(item.quote) > MAX_SUPPORTING_EXCERPT_CHARACTERS
            or item.quote not in available[item.citation_id].text
        ):
            if item.citation_id in declared:
                code = GroundingErrorCode.DUPLICATE_CITATION
            elif item.citation_id not in available:
                code = GroundingErrorCode.UNKNOWN_CITATION
            elif not item.quote or len(item.quote) > MAX_SUPPORTING_EXCERPT_CHARACTERS:
                code = GroundingErrorCode.QUOTE_MISMATCH
            else:
                code = GroundingErrorCode.QUOTE_NOT_IN_EVIDENCE
            raise GroundingValidationError(code)
        declared[item.citation_id] = item.quote
    marker_ids = [int(value) for value in _MARKER.findall(answer.answer)]
    if not marker_ids or set(marker_ids) != set(declared):
        code = (
            GroundingErrorCode.MISSING_CITATION
            if not marker_ids
            else GroundingErrorCode.MARKER_CITATION_MISMATCH
        )
        raise GroundingValidationError(code)
    mapping: dict[int, int] = {}
    for identifier in marker_ids:
        if identifier not in mapping:
            mapping[identifier] = len(mapping) + 1
    rendered = _MARKER.sub(lambda found: f"[{mapping[int(found.group(1))]}]", answer.answer)
    citations = tuple(
        Citation(
            public,
            available[internal].chunk_id,
            available[internal].document_id,
            available[internal].title,
            available[internal].source_filename,
            available[internal].locator,
            available[internal].citation_label,
            declared[internal],
            available[internal].score,
        )
        for internal, public in mapping.items()
    )
    if len(rendered.strip()) > 25 and not _MARKER.search(rendered):
        raise GroundingValidationError(GroundingErrorCode.UNSUPPORTED_CLAIM)
    return rendered, citations
