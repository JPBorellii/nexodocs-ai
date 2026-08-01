"""Tests for the public package contract."""

import nexodocs_ai


def test_package_can_be_imported_with_expected_version() -> None:
    """The package exposes its initial public version."""
    assert nexodocs_ai.__version__ == "0.1.0"
