"""Injected OpenAI boundary and deterministic offline answer provider."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol, cast

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

from .config import RagConfig
from .models import (
    AnswerGenerationRequest,
    GeneratedAnswer,
    GeneratedCitationReference,
    GenerationUsage,
    ProviderError,
    ProviderRefusalError,
    RagError,
)


class ResponsesEndpoint(Protocol):
    def create(self, **kwargs: object) -> object: ...


class ResponsesClient(Protocol):
    @property
    def responses(self) -> ResponsesEndpoint: ...


def generated_answer_schema() -> dict[str, object]:
    path = (
        Path(__file__).resolve().parents[3]
        / "knowledge_base"
        / "metadata"
        / "rag-generated-answer.schema.json"
    )
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def openai_generated_answer_schema() -> dict[str, object]:
    """Project the canonical response contract to OpenAI Structured Outputs."""

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

    return cast(dict[str, object], normalize(deepcopy(generated_answer_schema())))


def _from_json(value: str) -> GeneratedAnswer:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ProviderError("JSON inválido do provedor") from exc
    errors = list(Draft202012Validator(cast(Any, generated_answer_schema())).iter_errors(raw))  # pyright: ignore[reportUnknownMemberType] - jsonschema is a dynamic boundary.
    if errors or not isinstance(raw, dict):
        raise ProviderError("Schema inválido do provedor")
    raw_object = cast(dict[str, object], raw)
    citations = raw_object.get("citations")
    if not isinstance(citations, list):
        raise ProviderError("Citações inválidas")
    return GeneratedAnswer(
        str(raw_object["answer"]),
        tuple(
            GeneratedCitationReference(
                int(str(cast(dict[str, object], item)["citation_id"])),
                str(cast(dict[str, object], item)["quote"]),
            )
            for item in cast(list[object], citations)
        ),
    )


class OpenAIAnswerProvider:
    provider_name = "openai"

    def __init__(self, config: RagConfig, client: ResponsesClient | None = None) -> None:
        if config.app_env == "test":
            raise RagError("OpenAI não é permitido em APP_ENV=test")
        if not config.openai_answer_model:
            raise RagError("OPENAI_ANSWER_MODEL é obrigatório")
        if not config.openai_api_key and client is None:
            raise RagError("OPENAI_API_KEY é obrigatória")
        self.model_identifier = config.openai_answer_model
        self._max_output_tokens = config.openai_answer_max_output_tokens
        self._transport_retries = config.openai_answer_max_retries
        self._client: ResponsesClient = client or cast(
            ResponsesClient,
            OpenAI(
                api_key=config.openai_api_key,
                timeout=config.openai_answer_timeout_seconds,
                max_retries=config.openai_answer_max_retries,
            ),
        )

    def _usage(
        self, response: object, request: AnswerGenerationRequest, refusal: bool
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

    def _failed_usage(self, request: AnswerGenerationRequest) -> GenerationUsage:
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

    def generate(self, request: AnswerGenerationRequest) -> GeneratedAnswer:
        try:
            response = self._client.responses.create(
                model=self.model_identifier,
                instructions=request.system_prompt,
                input=request.user_prompt,
                max_output_tokens=self._max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "rag_generated_answer",
                        "strict": True,
                        "schema": openai_generated_answer_schema(),
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
        ) as exc:
            raise ProviderError(
                "Falha controlada do provedor", self._failed_usage(request)
            ) from exc
        refusal = getattr(response, "refusal", None)
        usage = self._usage(response, request, bool(refusal))
        if refusal:
            raise ProviderRefusalError("Recusa do provedor", usage)
        output = getattr(response, "output_text", None)
        if not isinstance(output, str) or not output.strip():
            raise ProviderError("Resposta vazia do provedor", usage)
        try:
            answer = _from_json(output)
        except ProviderError as exc:
            raise ProviderError(str(exc), usage) from exc
        return GeneratedAnswer(answer.answer, answer.citations, usage)


class DeterministicFakeAnswerProvider:
    provider_name = "deterministic-fake"
    model_identifier = "grounded-extract-test-v1"

    def __init__(self, config: RagConfig) -> None:
        if config.app_env != "test":
            raise RagError("Provedor fake é permitido somente em APP_ENV=test")

    def generate(self, request: AnswerGenerationRequest) -> GeneratedAnswer:
        if not request.evidence_blocks:
            raise ProviderError("Evidência obrigatória")
        selected = request.evidence_blocks[:2]
        citations: list[GeneratedCitationReference] = []
        fragments: list[str] = []
        for block in selected:
            sentence = next(
                (item.strip() for item in re.split(r"(?<=[.!?])\s+", block.text) if item.strip()),
                "",
            )
            if not sentence:
                raise ProviderError("Evidência sem frase")
            quote = sentence[:500].rsplit(" ", 1)[0] if len(sentence) > 500 else sentence
            citations.append(GeneratedCitationReference(block.evidence_id, quote))
            fragments.append(f"{quote} [{block.evidence_id}]")
        answer = " ".join(fragments)
        usage = GenerationUsage(
            self.provider_name,
            self.model_identifier,
            None,
            None,
            None,
            None,
            None,
            0,
            0,
            False,
            request.attempt_number,
            True,
        )
        return GeneratedAnswer(answer[: request.max_answer_characters], tuple(citations), usage)


def create_answer_provider(
    config: RagConfig, client: ResponsesClient | None = None
) -> OpenAIAnswerProvider | DeterministicFakeAnswerProvider:
    if config.answer_provider == "fake":
        return DeterministicFakeAnswerProvider(config)
    return OpenAIAnswerProvider(config, client)
