"""Isolated OpenAI boundary for semantic answerability decisions."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from jsonschema import Draft202012Validator
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from nexodocs_ai.observability.safety import safe_opaque_identifier

from .answerability_prompts import ANSWERABILITY_SYSTEM_PROMPT, render_answerability_input
from .models import (
    AnswerabilityDecision,
    AnswerabilityProviderError,
    AnswerabilityProviderRefusalError,
    AnswerabilityRequest,
    AnswerabilitySupport,
    EvidenceBlock,
    GenerationUsage,
    RagError,
)

_MISSING_RESPONSE = object()


class AnswerabilityResponsesEndpoint(Protocol):
    def create(self, **kwargs: object) -> object: ...


class AnswerabilityResponsesClient(Protocol):
    @property
    def responses(self) -> AnswerabilityResponsesEndpoint: ...


def answerability_decision_schema() -> dict[str, object]:
    """Load the canonical closed answerability decision schema."""
    path = (
        Path(__file__).resolve().parents[3]
        / "knowledge_base"
        / "metadata"
        / "rag-answerability-decision.schema.json"
    )
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def openai_answerability_decision_schema() -> dict[str, object]:
    """Project the canonical contract to strict OpenAI Structured Outputs."""

    def normalize(value: object) -> object:
        if isinstance(value, dict):
            mapping = cast(dict[str, object], value)
            return {
                key: normalize(nested)
                for key, nested in mapping.items()
                if key not in {"$schema", "uniqueItems"}
            }
        if isinstance(value, list):
            return [normalize(item) for item in cast(list[object], value)]
        return value

    return cast(dict[str, object], normalize(deepcopy(answerability_decision_schema())))


def _evidence_by_id(evidence_blocks: tuple[EvidenceBlock, ...]) -> dict[int, EvidenceBlock]:
    by_id = {block.evidence_id: block for block in evidence_blocks}
    if len(by_id) != len(evidence_blocks):
        raise AnswerabilityProviderError("Identificadores de evidência de entrada duplicados")
    return by_id


def _response_contains_refusal(response: object) -> bool:
    """Detect a Responses content refusal without reading or exposing its text."""
    output = getattr(response, "output", None)
    if isinstance(output, (list, tuple)):
        output_items = cast(list[object] | tuple[object, ...], output)
        for item in output_items:
            content = getattr(item, "content", None)
            if not isinstance(content, (list, tuple)):
                continue
            content_items = cast(list[object] | tuple[object, ...], content)
            if any(getattr(part, "type", None) == "refusal" for part in content_items):
                return True
    top_level = getattr(response, "refusal", None)
    return top_level is True or (isinstance(top_level, str) and bool(top_level))


def _parse_and_validate_output(
    value: str, evidence_blocks: tuple[EvidenceBlock, ...]
) -> AnswerabilityDecision:
    """Validate shape and literal quote provenance, not semantic entailment."""
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        raise AnswerabilityProviderError("JSON inválido do provedor") from None
    validator = Draft202012Validator(cast(Any, answerability_decision_schema()))
    errors = list(
        validator.iter_errors(raw)  # pyright: ignore[reportUnknownMemberType]
    )
    if errors or not isinstance(raw, dict):
        raise AnswerabilityProviderError("Schema inválido do provedor")

    raw_object = cast(dict[str, object], raw)
    raw_decision = raw_object["decision"]
    if raw_decision not in {"answerable", "insufficient"}:
        raise AnswerabilityProviderError("Decisão inválida do provedor")
    decision = cast(Literal["answerable", "insufficient"], raw_decision)
    evidence_by_id = _evidence_by_id(evidence_blocks)
    raw_support = cast(list[object], raw_object["supporting_evidence"])
    supports: list[AnswerabilitySupport] = []
    seen_ids: set[int] = set()
    for item in raw_support:
        support = cast(dict[str, object], item)
        evidence_id = cast(int, support["evidence_id"])
        quote = cast(str, support["quote"])
        if evidence_id not in evidence_by_id:
            raise AnswerabilityProviderError("Referência a evidência desconhecida")
        if evidence_id in seen_ids:
            raise AnswerabilityProviderError("Referência a evidência duplicada")
        if not quote.strip():
            raise AnswerabilityProviderError("Trecho de evidência vazio")
        if quote not in evidence_by_id[evidence_id].text:
            raise AnswerabilityProviderError("Trecho não pertence literalmente à evidência")
        seen_ids.add(evidence_id)
        supports.append(AnswerabilitySupport(evidence_id, quote))

    if decision == "answerable" and not supports:
        raise AnswerabilityProviderError("Decisão answerable exige evidência de suporte")
    if decision == "insufficient" and supports:
        raise AnswerabilityProviderError("Decisão insufficient não aceita evidência de suporte")
    return AnswerabilityDecision(decision, tuple(supports))


class OpenAIAnswerabilityProvider:
    """Strict Structured Outputs provider, isolated from final answer generation."""

    provider_name = "openai"

    def __init__(
        self,
        *,
        model_identifier: str,
        max_output_tokens: int,
        timeout_seconds: float,
        transport_retries: int,
        api_key: str | None = None,
        client: AnswerabilityResponsesClient | None = None,
    ) -> None:
        if not model_identifier.strip():
            raise RagError("Identificador do modelo de answerability é obrigatório")
        if max_output_tokens <= 0:
            raise RagError("Limite de saída de answerability deve ser positivo")
        if timeout_seconds <= 0:
            raise RagError("Timeout de answerability deve ser positivo")
        if transport_retries < 0:
            raise RagError("Tentativas de transporte de answerability não podem ser negativas")
        if client is None and not api_key:
            raise RagError("Chave explícita ou cliente injetado é obrigatório")
        self.model_identifier = model_identifier
        self._max_output_tokens = max_output_tokens
        self._transport_retries = transport_retries
        self._client: AnswerabilityResponsesClient = (
            client
            if client is not None
            else cast(
                AnswerabilityResponsesClient,
                OpenAI(
                    api_key=api_key,
                    timeout=timeout_seconds,
                    max_retries=transport_retries,
                ),
            )
        )

    def _usage(
        self, response: object, request: AnswerabilityRequest, refusal: bool
    ) -> GenerationUsage:
        raw = getattr(response, "usage", None)

        def integer(source: object, name: str) -> int | None:
            value = getattr(source, name, None)
            return value if isinstance(value, int) and value >= 0 else None

        input_tokens = integer(raw, "input_tokens")
        output_tokens = integer(raw, "output_tokens")
        total_tokens = integer(raw, "total_tokens")
        details = getattr(raw, "input_tokens_details", None)
        cached_input_tokens = integer(details, "cached_tokens")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        attempts_observable = self._transport_retries == 0
        return GenerationUsage(
            self.provider_name,
            self.model_identifier,
            input_tokens,
            cached_input_tokens,
            output_tokens,
            total_tokens,
            safe_opaque_identifier(
                getattr(response, "_request_id", None) or getattr(response, "request_id", None)
            ),
            1,
            1 if attempts_observable else None,
            refusal,
            request.attempt_number,
            attempts_observable,
        )

    def _failed_usage(self, request: AnswerabilityRequest) -> GenerationUsage:
        """Record only safe, observable metadata for a failed SDK request."""
        attempts_observable = self._transport_retries == 0
        return GenerationUsage(
            self.provider_name,
            self.model_identifier,
            None,
            None,
            None,
            None,
            None,
            1,
            1 if attempts_observable else None,
            False,
            request.attempt_number,
            attempts_observable,
        )

    def assess(self, request: AnswerabilityRequest) -> AnswerabilityDecision:
        """Request one decision and fail closed on every provider contract violation."""
        _evidence_by_id(request.evidence_blocks)
        response: object = _MISSING_RESPONSE
        try:
            response = self._client.responses.create(
                model=self.model_identifier,
                instructions=ANSWERABILITY_SYSTEM_PROMPT,
                input=render_answerability_input(request),
                max_output_tokens=self._max_output_tokens,
                store=False,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "rag_answerability_decision",
                        "strict": True,
                        "schema": openai_answerability_decision_schema(),
                    }
                },
            )
        except (
            BadRequestError,
            AuthenticationError,
            PermissionDeniedError,
            NotFoundError,
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
            APIStatusError,
        ):
            pass
        if response is _MISSING_RESPONSE:
            raise AnswerabilityProviderError(
                "Falha controlada do provedor de answerability", self._failed_usage(request)
            )
        refusal_detected = _response_contains_refusal(response)
        usage = self._usage(response, request, refusal_detected)
        if refusal_detected:
            raise AnswerabilityProviderRefusalError("Recusa do provedor de answerability", usage)
        output = getattr(response, "output_text", None)
        if not isinstance(output, str) or not output.strip():
            raise AnswerabilityProviderError("Resposta vazia do provedor de answerability", usage)
        try:
            decision = _parse_and_validate_output(output, request.evidence_blocks)
        except AnswerabilityProviderError:
            raise AnswerabilityProviderError(
                "Saída inválida do provedor de answerability", usage
            ) from None
        return AnswerabilityDecision(decision.decision, decision.supporting_evidence, usage)
