"""Validate the offline OpenAI Structured Outputs projection for generated answers."""

from __future__ import annotations

from typing import cast

from _project_bootstrap import bootstrap_project

_PROHIBITED_KEYWORDS = frozenset({"$schema", "uniqueItems"})


class OpenAISchemaValidationError(RuntimeError):
    """Raised when the provider-facing projection changes its closed contract."""


def _validate_objects(value: object) -> None:
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        if any(keyword in mapping for keyword in _PROHIBITED_KEYWORDS):
            raise OpenAISchemaValidationError("prohibited_keyword")
        if mapping.get("type") == "object":
            properties = mapping.get("properties")
            required = mapping.get("required")
            if (
                mapping.get("additionalProperties") is not False
                or not isinstance(properties, dict)
                or not isinstance(required, list)
            ):
                raise OpenAISchemaValidationError("closed_object_required")
            property_mapping = cast(dict[str, object], properties)
            required_values = cast(list[object], required)
            if not all(isinstance(item, str) for item in required_values) or set(
                property_mapping
            ) != set(cast(list[str], required_values)):
                raise OpenAISchemaValidationError("closed_object_required")
        for nested in mapping.values():
            _validate_objects(nested)
    elif isinstance(value, list):
        for nested in cast(list[object], value):
            _validate_objects(nested)


def validate_schema() -> None:
    """Validate provider compatibility and equivalence to the canonical schema."""
    from nexodocs_ai.rag.answer_provider import (
        generated_answer_schema,
        openai_generated_answer_schema,
    )

    canonical, provider = generated_answer_schema(), openai_generated_answer_schema()
    if provider.get("type") != "object":
        raise OpenAISchemaValidationError("root_object_required")
    _validate_objects(provider)
    properties = cast(dict[str, object], provider.get("properties"))
    if set(properties) != {"answer", "citations"}:
        raise OpenAISchemaValidationError("response_properties_invalid")
    citations = cast(dict[str, object], properties["citations"])
    citation_items = cast(dict[str, object], citations["items"])
    if set(cast(dict[str, object], citation_items["properties"])) != {"citation_id", "quote"}:
        raise OpenAISchemaValidationError("citation_properties_invalid")

    def remove_prohibited(value: object) -> object:
        if isinstance(value, dict):
            mapping = cast(dict[str, object], value)
            return {
                key: remove_prohibited(nested)
                for key, nested in mapping.items()
                if key not in _PROHIBITED_KEYWORDS
            }
        if isinstance(value, list):
            return [remove_prohibited(item) for item in cast(list[object], value)]
        return value

    if provider != remove_prohibited(canonical):
        raise OpenAISchemaValidationError("canonical_equivalence_invalid")


def main() -> int:
    """Run the provider schema validation without calling OpenAI or the internet."""
    bootstrap_project()
    try:
        validate_schema()
    except OpenAISchemaValidationError:
        print("OpenAI Structured Outputs schema validation failed.")
        return 1
    print("OpenAI Structured Outputs schema validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
