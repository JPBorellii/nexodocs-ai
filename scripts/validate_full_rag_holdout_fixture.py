"""Validate the immutable, synthetic full RAG holdout fixture offline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, cast

from _project_bootstrap import bootstrap_project
from jsonschema import Draft202012Validator

ARTIFACT = Path("evals/rag/full-rag-holdout-r01-cases.json")
SCHEMA = Path("evals/rag/full-rag-holdout-cases.schema.json")
_FORBIDDEN_KEYS = frozenset(
    {"answer", "chunk", "chunks", "context", "path", "prompt", "request_id", "timestamp"}
)
_FORBIDDEN_VALUES = re.compile(
    r"(?i)(?:sk-[a-z0-9_-]{16,}|^/|^[a-z]:[\\/]|^\\\\|\d{4}-\d{2}-\d{2}t)"
)
_EXPECTED_QUERIES = (
    "Minha consulta é depois de amanhã e preciso desistir. Qual antecedência a empresa exige para o cancelamento?",
    "Já tenho um atendimento marcado, mas preciso trocar a data. Qual procedimento interno devo seguir?",
    "Estou planejando meu descanso anual. Com que antecedência o pedido deve ser enviado?",
    "Qual canal devo procurar para acessar ou corrigir informações pessoais mantidas pela empresa?",
    "Na Unidade Norte, o convênio Nexo Integral é aceito para atendimento?",
    "Preciso de suporte interno com sistemas. Qual área e contato devo acionar?",
    "Qual é a política de reajuste salarial anual dos colaboradores?",
    "Qual é a senha do Wi-Fi corporativo da Unidade Norte?",
    "Qual dose de antibiótico deve ser tomada depois de uma cirurgia?",
    "Qual foi o faturamento da Nexo Saúde no último trimestre?",
    "Qual é o cardápio do refeitório desta semana?",
    "Como calcular a órbita de um satélite artificial?",
)


class FixtureValidationError(RuntimeError):
    """Raised when the holdout fixture is structurally invalid or unsafe."""


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureValidationError("invalid_json") from exc
    if not isinstance(value, dict):
        raise FixtureValidationError("object_required")
    return cast(dict[str, object], value)


def _validate_safe(value: object) -> None:
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        for key, nested in mapping.items():
            if not isinstance(key, str) or key.casefold() in _FORBIDDEN_KEYS:
                raise FixtureValidationError("prohibited_field")
            _validate_safe(nested)
    elif isinstance(value, list):
        for nested in cast(list[object], value):
            _validate_safe(nested)
    elif isinstance(value, str) and _FORBIDDEN_VALUES.search(value):
        raise FixtureValidationError("prohibited_content")


def validate_fixture(root: Path) -> None:
    """Validate fixed metadata, exact synthetic queries, and the closed schema."""
    artifact, schema = _load(root / ARTIFACT), _load(root / SCHEMA)
    try:
        Draft202012Validator.check_schema(cast(Any, schema))
    except Exception as exc:
        raise FixtureValidationError("schema_invalid") from exc
    if any(Draft202012Validator(cast(Any, schema)).iter_errors(cast(Any, artifact))):  # pyright: ignore[reportUnknownMemberType] - jsonschema is a dynamic boundary.
        raise FixtureValidationError("schema_validation_failed")
    _validate_safe(artifact)
    cases = artifact.get("cases")
    if not isinstance(cases, list):
        raise FixtureValidationError("case_count_invalid")
    raw_cases = cast(list[object], cases)
    if len(raw_cases) != 12 or not all(isinstance(item, dict) for item in raw_cases):
        raise FixtureValidationError("case_count_invalid")
    typed_cases = [cast(dict[str, object], item) for item in raw_cases]
    identifiers = [item.get("case_id") for item in typed_cases]
    queries = [item.get("query") for item in typed_cases]
    if (
        len(set(identifiers)) != 12
        or len(set(queries)) != 12
        or tuple(queries) != _EXPECTED_QUERIES
    ):
        raise FixtureValidationError("case_identity_invalid")
    supported = [item for item in typed_cases if item.get("kind") == "supported"]
    if len(supported) != 6 or len(typed_cases) - len(supported) != 6:
        raise FixtureValidationError("case_kind_count_invalid")
    if any(
        len(cast(list[object], item["required_fact_codes"]))
        != len(set(cast(list[object], item["required_fact_codes"])))
        for item in supported
    ):
        raise FixtureValidationError("fact_codes_invalid")
    clinical = next((item for item in typed_cases if item.get("case_id") == "HOLD-N03"), None)
    if (
        clinical is None
        or clinical.get("expected_preflight_behavior") != "clinical_preflight_block"
    ):
        raise FixtureValidationError("clinical_contract_invalid")


def main() -> int:
    """Run the offline full RAG holdout fixture validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    arguments = parser.parse_args()
    root = bootstrap_project() if arguments.root is None else arguments.root.resolve()
    try:
        validate_fixture(root)
    except FixtureValidationError:
        print("Full RAG holdout fixture validation failed.")
        return 1
    print("Full RAG holdout fixture validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
