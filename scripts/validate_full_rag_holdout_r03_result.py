"""CLI for offline validation of the sanitized full-RAG holdout R03 result."""

from __future__ import annotations

import argparse
from pathlib import Path

from _project_bootstrap import bootstrap_project


def build_parser() -> argparse.ArgumentParser:
    """Build the offline result-validator CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--report-root", type=Path)
    return parser


def main() -> int:
    """Run the R03 result validator without invoking any external service."""
    arguments = build_parser().parse_args()
    root = bootstrap_project() if arguments.root is None else arguments.root.resolve()
    report_root = None if arguments.report_root is None else arguments.report_root.resolve()

    from nexodocs_ai.rag.full_rag_holdout_r03 import HoldoutHarnessError
    from nexodocs_ai.rag.full_rag_holdout_r03_result import (
        R03ResultValidationError,
        validate_result,
    )

    try:
        validate_result(root, report_root)
    except HoldoutHarnessError, OSError, R03ResultValidationError:
        print("Full RAG holdout R03 result validation failed.")
        return 1
    print("Full RAG holdout R03 result validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
