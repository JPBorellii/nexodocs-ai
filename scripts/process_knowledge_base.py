"""Generate or check deterministic processed knowledge-base artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nexodocs_ai.processing.models import ProcessingError
from nexodocs_ai.processing.pipeline import check, generate


def main() -> int:
    """Run the requested processing mode."""
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.write:
            generate(output_dir=args.output_dir)
        else:
            check(output_dir=args.output_dir)
    except ProcessingError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
