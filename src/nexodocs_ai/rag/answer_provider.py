"""Injected OpenAI boundary and deterministic offline answer provider."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol, cast

from jsonschema import Draft202012Validator
from openai import OpenAI

from .config import RagConfig
from .models import (
    AnswerGenerationRequest,
    GeneratedAnswer,
    GeneratedCitationReference,
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
        self._client: ResponsesClient = client or cast(
            ResponsesClient,
            OpenAI(
                api_key=config.openai_api_key,
                timeout=config.openai_answer_timeout_seconds,
                max_retries=config.openai_answer_max_retries,
            ),
        )

    def generate(self, request: AnswerGenerationRequest) -> GeneratedAnswer:
        response = self._client.responses.create(
            model=self.model_identifier,
            instructions=request.system_prompt,
            input=request.user_prompt,
            max_output_tokens=request.max_answer_characters,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "rag_generated_answer",
                    "strict": True,
                    "schema": generated_answer_schema(),
                }
            },
        )
        refusal = getattr(response, "refusal", None)
        if refusal:
            raise ProviderRefusalError("Recusa do provedor")
        output = getattr(response, "output_text", None)
        if not isinstance(output, str) or not output.strip():
            raise ProviderError("Resposta vazia do provedor")
        answer = _from_json(output)
        usage = getattr(response, "usage", None)
        return GeneratedAnswer(
            answer.answer,
            answer.citations,
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
        )


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
        return GeneratedAnswer(answer[: request.max_answer_characters], tuple(citations))


def create_answer_provider(
    config: RagConfig, client: ResponsesClient | None = None
) -> OpenAIAnswerProvider | DeterministicFakeAnswerProvider:
    if config.answer_provider == "fake":
        return DeterministicFakeAnswerProvider(config)
    return OpenAIAnswerProvider(config, client)
