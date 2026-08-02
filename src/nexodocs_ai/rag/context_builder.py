"""Deterministic context selection that counts the rendered data contract."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from nexodocs_ai.retrieval.models import RetrievalResult

from .models import ContextBuildResult, DiscardedEvidence, EvidenceBlock


def serialize_evidence(blocks: Sequence[EvidenceBlock]) -> str:
    """Serialize evidence as untrusted JSON data in a stable order."""
    payload = [
        {
            "evidence_id": block.evidence_id,
            "chunk_id": block.chunk_id,
            "document_id": block.document_id,
            "title": block.title,
            "source_filename": block.source_filename,
            "locator": block.locator,
            "citation_label": block.citation_label,
            "score": block.score,
            "text": block.text,
            "text_sha256": block.text_sha256,
        }
        for block in blocks
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ContextBuilder:
    """Keep complete ranked chunks within explicit, rendered-character limits."""

    def __init__(self, max_characters: int, max_per_document: int, max_chunks: int) -> None:
        self.max_characters, self.max_per_document, self.max_chunks = (
            max_characters,
            max_per_document,
            max_chunks,
        )

    def build(self, results: Sequence[RetrievalResult]) -> ContextBuildResult:
        blocks: list[EvidenceBlock] = []
        discarded: list[DiscardedEvidence] = []
        seen: set[str] = set()
        document_counts: dict[str, int] = {}
        for result in results:
            document_id = str(result.metadata.get("document_id", ""))
            title, text_hash = (
                str(result.metadata.get("title", "")),
                str(result.metadata.get("text_sha256", "")),
            )
            if (
                not result.chunk_id
                or not document_id
                or not result.text.strip()
                or not title
                or not text_hash
            ):
                discarded.append(DiscardedEvidence(result.chunk_id, "invalid_evidence"))
                continue
            if result.chunk_id in seen:
                discarded.append(DiscardedEvidence(result.chunk_id, "duplicate_chunk"))
                continue
            if len(blocks) >= self.max_chunks:
                discarded.append(DiscardedEvidence(result.chunk_id, "chunk_limit"))
                continue
            if document_counts.get(document_id, 0) >= self.max_per_document:
                discarded.append(DiscardedEvidence(result.chunk_id, "document_limit"))
                continue
            block = EvidenceBlock(
                len(blocks) + 1,
                result.chunk_id,
                document_id,
                title,
                result.source,
                result.locator,
                result.citation_label,
                result.score,
                result.text,
                text_hash,
            )
            candidate = [*blocks, block]
            if len(serialize_evidence(candidate)) > self.max_characters:
                discarded.append(DiscardedEvidence(result.chunk_id, "context_budget"))
                continue
            blocks.append(block)
            seen.add(result.chunk_id)
            document_counts[document_id] = document_counts.get(document_id, 0) + 1
        rendered = serialize_evidence(blocks)
        return ContextBuildResult(
            tuple(blocks),
            tuple(item.chunk_id for item in blocks),
            tuple(discarded),
            len(rendered),
            sum(len(item.text.split()) for item in blocks),
            hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        )
