"""Generate or check the fictitious knowledge base."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nexodocs_ai.knowledge_base.generator import check, generate
from nexodocs_ai.knowledge_base.models import KnowledgeBaseError


def main() -> int:
    """Run the selected deterministic generation mode."""
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
    except KnowledgeBaseError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
