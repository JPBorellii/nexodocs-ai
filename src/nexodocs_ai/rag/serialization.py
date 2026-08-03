"""Privacy-aware serialization for the manual RAG CLI."""

from __future__ import annotations

from dataclasses import asdict

from .grounding_diagnostics import validate_safe_error_code
from .models import RagResponse


def cli_response_data(response: RagResponse) -> dict[str, object]:
    """Return a controlled payload, suppressing content on grounding failure."""
    if response.status == "grounding_failed":
        validate_safe_error_code(response.status, response.reason_code)
        return {"safe_error_code": response.reason_code, "status": response.status}
    return asdict(response)


def cli_exit_code(response: RagResponse) -> int:
    """Preserve the established CLI success/failure contract."""
    return 0 if response.status in {"answered", "no_evidence"} else 1


def privacy_safe_report_required(response: RagResponse) -> bool:
    """Require the identifier-free report boundary after grounding rejection."""
    return response.status == "grounding_failed"
