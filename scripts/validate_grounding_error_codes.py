"""Validate the closed sanitized grounding-error taxonomy."""

from __future__ import annotations

from _project_bootstrap import bootstrap_project


def main() -> int:
    """Check the exact stable code set without external services."""
    bootstrap_project()
    from nexodocs_ai.rag.grounding_diagnostics import (
        GROUNDING_SAFE_ERROR_CODES,
        GroundingErrorCode,
        validate_safe_error_code,
    )

    expected = frozenset(code.value for code in GroundingErrorCode)
    if GROUNDING_SAFE_ERROR_CODES != expected:
        print("Sanitized grounding error code validation failed.")
        return 1
    for code in expected:
        validate_safe_error_code("grounding_failed", code)
    try:
        validate_safe_error_code("grounding_failed", "raw provider message")
    except ValueError:
        print("Sanitized grounding error code validation passed.")
        return 0
    print("Sanitized grounding error code validation failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
