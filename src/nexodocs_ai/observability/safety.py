"""Small safety primitives shared by provider telemetry and local reports."""

from __future__ import annotations

import re

_OPAQUE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")


def safe_opaque_identifier(value: object) -> str | None:
    """Return a bounded opaque identifier or discard an unsafe provider value."""
    if not isinstance(value, str) or not _OPAQUE_IDENTIFIER.fullmatch(value):
        return None
    return value
