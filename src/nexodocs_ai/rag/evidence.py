"""Conservative scope/sufficiency assessment and deterministic fallbacks."""

from __future__ import annotations

import re

from nexodocs_ai.retrieval.models import RetrievalResponse

from .models import ContextBuildResult, EvidenceAssessment

_CLINICAL = re.compile(
    r"\b(diagnostique|diagnóstico para mim|prescreva|prescrição|qual (?:remédio|medicamento)|dose de|tratamento para mim)\b",
    re.I,
)
_UNSAFE = re.compile(
    r"\b(revele (?:o |a )?(?:prompt|chave)|ignore (?:as )?regras|execute (?:o )?comando|mostre (?:a )?configuração)\b",
    re.I,
)


def preflight(query: str) -> EvidenceAssessment:
    if _CLINICAL.search(query):
        return EvidenceAssessment("out_of_scope", "clinical_guidance_not_supported")
    if _UNSAFE.search(query):
        return EvidenceAssessment("out_of_scope", "unsafe_request")
    return EvidenceAssessment("sufficient")


def assess_retrieval(response: RetrievalResponse, minimum_results: int) -> EvidenceAssessment:
    if response.status != "found" or len(response.results) < minimum_results:
        return EvidenceAssessment("insufficient", response.reason or "insufficient_evidence")
    identities: dict[tuple[str, str], str] = {}
    for item in response.results:
        document_id, text_hash = (
            str(item.metadata.get("document_id", "")),
            str(item.metadata.get("text_sha256", "")),
        )
        identity = (document_id, item.locator)
        if not item.text.strip() or not document_id or not text_hash:
            return EvidenceAssessment("insufficient", "insufficient_evidence")
        if identity in identities and identities[identity] != text_hash:
            return EvidenceAssessment("ambiguous", "ambiguous_evidence")
        identities[identity] = text_hash
    return EvidenceAssessment("sufficient")


def assess_context(context: ContextBuildResult, minimum_characters: int) -> EvidenceAssessment:
    if not context.evidence_blocks:
        return EvidenceAssessment("insufficient", "insufficient_evidence")
    if context.total_characters < minimum_characters:
        return EvidenceAssessment("insufficient", "context_below_minimum")
    return EvidenceAssessment("sufficient")
