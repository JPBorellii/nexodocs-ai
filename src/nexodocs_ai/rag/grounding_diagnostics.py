"""Closed, sanitized categories for deterministic grounding failures."""

from __future__ import annotations

from enum import StrEnum

from .models import ValidationError


class GroundingErrorCode(StrEnum):
    """Stable categories emitted by the current grounding validator."""

    SCHEMA_INVALID = "grounding_schema_invalid"
    ANSWER_TOO_LONG = "grounding_answer_too_long"
    UNSUPPORTED_CLAIM = "grounding_unsupported_claim"
    DUPLICATE_CITATION = "grounding_duplicate_citation"
    UNKNOWN_CITATION = "grounding_unknown_citation"
    QUOTE_MISMATCH = "grounding_quote_mismatch"
    QUOTE_NOT_IN_EVIDENCE = "grounding_quote_not_in_evidence"
    MISSING_CITATION = "grounding_missing_citation"
    MARKER_CITATION_MISMATCH = "grounding_marker_citation_mismatch"
    VALIDATION_FAILED = "grounding_validation_failed"


GROUNDING_SAFE_ERROR_CODES = frozenset(code.value for code in GroundingErrorCode)


class GroundingValidationError(ValidationError):
    """Grounding failure carrying only a closed, privacy-safe category."""

    def __init__(self, safe_error_code: GroundingErrorCode) -> None:
        super().__init__("Grounding validation failed")
        self.safe_error_code = safe_error_code.value


def safe_grounding_error_code(error: ValidationError) -> str:
    """Return a closed code without using exception text as a contract."""
    if isinstance(error, GroundingValidationError):
        return error.safe_error_code
    return GroundingErrorCode.VALIDATION_FAILED.value


def validate_safe_error_code(status: str, safe_error_code: str | None) -> None:
    """Reject unknown codes specifically at the grounding-failure boundary."""
    if status == "grounding_failed" and safe_error_code not in GROUNDING_SAFE_ERROR_CODES:
        raise ValueError("Invalid sanitized grounding error code")
