"""End-to-end CLI coverage for the opt-in D04 boundary using only fake providers."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Literal, Protocol, cast

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))


class AnswerCliModule(Protocol):
    bootstrap_project: Callable[[], Path]

    def main(self) -> int: ...


class D04ValidatorModule(Protocol):
    def validate_d04(
        self,
        root: Path,
        artifact: Path | None = None,
        *,
        usage_report: Path | None = None,
        expected_commit: str | None = None,
        schema_only: bool = False,
    ) -> dict[str, object] | None: ...


answer_cli = cast(AnswerCliModule, importlib.import_module("answer_question"))
d04_validator = cast(
    D04ValidatorModule, importlib.import_module("validate_grounding_diagnostic_d04")
)

from nexodocs_ai.observability import grounding_diagnostic_d04 as d04  # noqa: E402
from nexodocs_ai.rag.models import (  # noqa: E402
    EvidenceBlock,
    GenerationUsage,
    RagResponse,
    RagRunResult,
)
from nexodocs_ai.rag.sanitized_grounding import project_quote_failure  # noqa: E402
from nexodocs_ai.retrieval.models import EmbeddingRunUsage  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40
PRIVATE_SENTINELS = (
    "SENTINEL_PRIVATE_PATH",
    "SENTINEL_PRIVATE_QUESTION",
    "SENTINEL_PRIVATE_ANSWER",
    "SENTINEL_PRIVATE_QUOTE",
    "SENTINEL_PRIVATE_EVIDENCE",
    "SENTINEL_PRIVATE_PROMPT",
    "SENTINEL_PRIVATE_CITATION",
    "sk-proj-SENTINEL_SECRET_NOT_REAL",
)
RagStatus = Literal[
    "answered", "no_evidence", "invalid_request", "generation_failed", "grounding_failed"
]
Pathish = str | bytes | os.PathLike[str] | os.PathLike[bytes]


def _prepare_root(tmp_path: Path) -> None:
    schema = tmp_path / d04.ARTIFACT_SCHEMA
    schema.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / d04.ARTIFACT_SCHEMA, schema)
    for relative in d04.SOURCE_PATHS.values():
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)


def _usage() -> GenerationUsage:
    return GenerationUsage("fake", "fake-answer", 15, 0, 5, 20, None, 1, 1, False, 1, True)


def _run(status: str = "grounding_failed", reason: str | None = None) -> RagRunResult:
    retrieval = EmbeddingRunUsage("fake", "fake-embedding", 3, 1, 1, 1, 1, 1, 10, 10, True, ())
    answer_usage = None if status == "no_evidence" else _usage()
    response = RagResponse(
        "1.0",
        cast(RagStatus, status),
        "SENTINEL_PRIVATE_QUESTION",
        (),
        answer="Synthetic safe answer" if status == "answered" else None,
        reason_code=reason,
        message="Synthetic safe fallback" if status != "answered" else None,
    )
    signals = None
    if status == "grounding_failed" and reason == "grounding_quote_not_in_evidence":
        evidence = EvidenceBlock(
            1,
            "synthetic-chunk",
            "synthetic-document",
            "Synthetic",
            "synthetic.txt",
            "section=synthetic",
            "Synthetic",
            0.9,
            "SENTINEL_PRIVATE_EVIDENCE",
            "b" * 64,
        )
        signals = (project_quote_failure("SENTINEL_PRIVATE_QUOTE", 1, (evidence,)),)
    return RagRunResult(
        response, retrieval, answer_usage, 0 if answer_usage is None else 1, signals
    )


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run: RagRunResult,
) -> tuple[type[object], dict[str, int]]:
    from nexodocs_ai.rag import answer_provider
    from nexodocs_ai.rag import config as rag_config
    from nexodocs_ai.rag import pipeline as pipeline_module
    from nexodocs_ai.retrieval import config as retrieval_config
    from nexodocs_ai.retrieval import embeddings, qdrant_store, retriever

    counters = {"load_config": 0}

    def fake_load_config() -> SimpleNamespace:
        counters["load_config"] += 1
        return SimpleNamespace(
            collection_name="synthetic-collection", top_k=5, max_top_k=10, max_per_document=2
        )

    def fake_load_rag_config() -> SimpleNamespace:
        return SimpleNamespace()

    def fake_embedding_provider(_config: object) -> SimpleNamespace:
        return SimpleNamespace(dimensions=3)

    def fake_client(_config: object) -> object:
        return object()

    def fake_answer_provider(_config: object) -> object:
        return object()

    def fake_commit(_root: Path) -> str:
        return COMMIT

    class FakeStore:
        def __init__(self, *_args: object) -> None:
            pass

        def validate_collection(self) -> None:
            pass

    class FakeRetriever:
        def __init__(self, *_args: object) -> None:
            pass

    class FakePipeline:
        instances: ClassVar[list[FakePipeline]] = []

        def __init__(self, *_args: object) -> None:
            self.provider = SimpleNamespace(provider_name="fake", model_identifier="fake-answer")
            self.calls = 0
            self.diagnostic_flags: list[bool] = []
            self.instances.append(self)

        def answer_with_usage(
            self, _request: object, *, sanitized_grounding_diagnostic: bool = False
        ) -> RagRunResult:
            self.calls += 1
            self.diagnostic_flags.append(sanitized_grounding_diagnostic)
            return run

        def answer(self, _request: object) -> RagResponse:
            self.calls += 1
            self.diagnostic_flags.append(False)
            return run.response

    monkeypatch.setattr(answer_cli, "bootstrap_project", lambda: tmp_path)
    monkeypatch.setattr(retrieval_config, "load_config", fake_load_config)
    monkeypatch.setattr(rag_config, "load_rag_config", fake_load_rag_config)
    monkeypatch.setattr(embeddings, "create_embedding_provider", fake_embedding_provider)
    monkeypatch.setattr(qdrant_store, "create_client", fake_client)
    monkeypatch.setattr(qdrant_store, "QdrantStore", FakeStore)
    monkeypatch.setattr(retriever, "Retriever", FakeRetriever)
    monkeypatch.setattr(pipeline_module, "RagPipeline", FakePipeline)
    monkeypatch.setattr(answer_provider, "create_answer_provider", fake_answer_provider)
    monkeypatch.setattr(d04, "_system_commit", fake_commit)
    return cast(type[object], FakePipeline), counters


def _arguments(usage: str, artifact: str) -> list[str]:
    return [
        "answer_question.py",
        "--query",
        "SENTINEL_PRIVATE_QUESTION",
        "--usage-report",
        usage,
        "--privacy-safe-usage-report",
        "--sanitized-grounding-diagnostic-report",
        artifact,
        "--sanitized-grounding-diagnostic-case-id",
        "HOLD-P02",
    ]


def test_cli_d04_success_writes_valid_sanitized_reports_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _prepare_root(tmp_path)
    pipeline_type, _ = _install_fakes(
        monkeypatch,
        tmp_path,
        _run("grounding_failed", "grounding_quote_not_in_evidence"),
    )
    usage_name = "data/run-reports/phase-6a-grounding-diagnostic-d04-p02-usage.json"
    artifact_name = "data/run-reports/phase-6a-grounding-diagnostic-d04-p02.json"
    monkeypatch.setattr(sys, "argv", _arguments(usage_name, artifact_name))

    assert answer_cli.main() == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "safe_error_code": "grounding_quote_not_in_evidence",
        "status": "grounding_failed",
    }
    usage_path, artifact_path = tmp_path / usage_name, tmp_path / artifact_name
    assert usage_path.exists() and artifact_path.exists()
    validated = d04_validator.validate_d04(
        tmp_path,
        artifact_path,
        usage_report=usage_path,
        expected_commit=COMMIT,
    )
    assert validated is not None
    assert validated["retrieval_logical_api_calls"] == 1
    assert validated["answer_logical_api_calls"] == 1
    assert validated["retrieval_physical_attempts"] == 1
    assert validated["answer_physical_attempts"] == 1
    instances = cast(list[object], getattr(pipeline_type, "instances"))
    assert len(instances) == 1
    instance = instances[0]
    assert getattr(instance, "calls") == 1
    assert getattr(instance, "diagnostic_flags") == [True]
    serialized = (
        captured.out
        + usage_path.read_text(encoding="utf-8")
        + artifact_path.read_text(encoding="utf-8")
    )
    assert not any(value in serialized for value in PRIVATE_SENTINELS)


@pytest.mark.parametrize(
    ("usage", "artifact"),
    (
        ("data/run-reports/same.json", "data/run-reports/same.json"),
        ("data/run-reports/./same.json", "data/run-reports/nested/../same.json"),
        ("data/run-reports/SAME.json", "data/run-reports/same.JSON"),
    ),
)
def test_cli_rejects_equivalent_paths_before_provider_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    usage: str,
    artifact: str,
) -> None:
    _prepare_root(tmp_path)
    pipeline_type, counters = _install_fakes(
        monkeypatch,
        tmp_path,
        _run("grounding_failed", "grounding_quote_not_in_evidence"),
    )
    monkeypatch.setattr(sys, "argv", _arguments(usage, artifact))
    assert answer_cli.main() == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["safe_error_code"] == "d04_destination_conflict"
    assert counters["load_config"] == 0
    assert getattr(pipeline_type, "instances") == []
    assert not list((tmp_path / "data/run-reports").glob("*.json"))


@pytest.mark.parametrize("existing_kind", ("file", "directory"))
def test_cli_rejects_existing_destination_before_provider_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    existing_kind: str,
) -> None:
    _prepare_root(tmp_path)
    pipeline_type, counters = _install_fakes(
        monkeypatch,
        tmp_path,
        _run("grounding_failed", "grounding_quote_not_in_evidence"),
    )
    destination = tmp_path / "data/run-reports/existing.json"
    destination.parent.mkdir(parents=True)
    if existing_kind == "file":
        destination.write_text("synthetic", encoding="utf-8")
    else:
        destination.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        _arguments("data/run-reports/usage.json", "data/run-reports/existing.json"),
    )
    assert answer_cli.main() == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    expected = "d04_destination_exists" if existing_kind == "file" else "d04_destination_invalid"
    assert json.loads(captured.out)["safe_error_code"] == expected
    assert counters["load_config"] == 0
    assert getattr(pipeline_type, "instances") == []


@pytest.mark.parametrize("failure", ("mkdir", "mkstemp", "link", "fsync", "cleanup"))
def test_cli_publication_failures_are_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    _prepare_root(tmp_path)
    _install_fakes(
        monkeypatch,
        tmp_path,
        _run("grounding_failed", "grounding_quote_not_in_evidence"),
    )
    usage = "data/run-reports/usage.json"
    artifact = "data/run-reports/restricted/d04.json"
    original_mkdir = Path.mkdir
    original_link = os.link
    original_fsync = os.fsync
    original_unlink = Path.unlink
    link_calls = 0
    fsync_calls = 0
    cleanup_failures = 0

    if failure == "mkdir":

        def fail_mkdir(path: Path, *args: object, **kwargs: object) -> None:
            if path.name == "restricted":
                raise PermissionError("SENTINEL_PRIVATE_PATH")
            original_mkdir(path, *args, **kwargs)  # pyright: ignore[reportArgumentType]

        monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    elif failure == "mkstemp":

        def fail_mkstemp(**_kwargs: object) -> tuple[int, str]:
            raise PermissionError("SENTINEL_PRIVATE_PATH")

        monkeypatch.setattr(d04.tempfile, "mkstemp", fail_mkstemp)
    elif failure == "link":

        def fail_second_link(source: Pathish, destination: Pathish) -> None:
            nonlocal link_calls
            link_calls += 1
            if link_calls == 2:
                raise OSError("SENTINEL_PRIVATE_PATH")
            original_link(source, destination)

        monkeypatch.setattr(os, "link", fail_second_link)
    elif failure == "fsync":

        def fail_second_fsync(descriptor: int) -> None:
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 2:
                raise OSError("SENTINEL_PRIVATE_PATH")
            original_fsync(descriptor)

        monkeypatch.setattr(os, "fsync", fail_second_fsync)
    else:

        def transient_cleanup(path: Path, *, missing_ok: bool = False) -> None:
            nonlocal cleanup_failures
            if path.name.startswith(".d04-") and cleanup_failures == 0:
                cleanup_failures += 1
                raise PermissionError("SENTINEL_PRIVATE_PATH")
            original_unlink(path, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", transient_cleanup)

    monkeypatch.setattr(sys, "argv", _arguments(usage, artifact))
    exit_code = answer_cli.main()
    captured = capsys.readouterr()
    assert captured.err == ""
    if failure == "cleanup":
        assert exit_code == 1
        assert json.loads(captured.out)["status"] == "grounding_failed"
        assert (tmp_path / artifact).exists()
    else:
        assert exit_code == 1
        assert json.loads(captured.out) == {
            "safe_error_code": "d04_artifact_write_failed",
            "status": "diagnostic_failed",
        }
        assert not (tmp_path / artifact).exists()
    assert "SENTINEL" not in captured.out + captured.err
    assert not list((tmp_path / "data/run-reports").rglob(".d04-*.tmp"))


@pytest.mark.parametrize(
    ("status", "reason"),
    (
        ("answered", None),
        ("no_evidence", "insufficient_evidence"),
        ("generation_failed", "provider_invalid_output"),
        ("generation_failed", "provider_refusal"),
        ("grounding_failed", "grounding_quote_mismatch"),
    ),
)
def test_cli_non_applicable_results_do_not_create_d04(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    reason: str | None,
) -> None:
    _prepare_root(tmp_path)
    _install_fakes(monkeypatch, tmp_path, _run(status, reason))
    artifact = "data/run-reports/not-applicable.json"
    monkeypatch.setattr(sys, "argv", _arguments("data/run-reports/usage.json", artifact))
    assert answer_cli.main() == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["safe_error_code"] == "d04_diagnostic_not_applicable"
    assert not (tmp_path / artifact).exists()


@pytest.mark.parametrize(
    ("extra", "expected"),
    (
        (
            ["--sanitized-grounding-diagnostic-report", "data/run-reports/d04.json"],
            "d04_case_id_invalid",
        ),
        (
            [
                "--sanitized-grounding-diagnostic-report",
                "data/run-reports/d04.json",
                "--sanitized-grounding-diagnostic-case-id",
                "HOLD-X99",
            ],
            "d04_case_id_invalid",
        ),
        (
            [
                "--sanitized-grounding-diagnostic-report",
                "data/run-reports/d04.json",
                "--sanitized-grounding-diagnostic-case-id",
                "HOLD-P02",
            ],
            "d04_usage_report_required",
        ),
        (
            [
                "--usage-report",
                "data/run-reports/usage.json",
                "--sanitized-grounding-diagnostic-report",
                "data/run-reports/d04.json",
                "--sanitized-grounding-diagnostic-case-id",
                "HOLD-P02",
            ],
            "d04_privacy_safe_required",
        ),
    ),
)
def test_cli_missing_or_invalid_d04_options_are_closed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    expected: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["answer_question.py", "--query", "synthetic", *extra])
    assert answer_cli.main() == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["safe_error_code"] == expected


def test_cli_without_d04_flag_keeps_normal_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _prepare_root(tmp_path)
    pipeline_type, _ = _install_fakes(monkeypatch, tmp_path, _run("answered"))
    monkeypatch.setattr(sys, "argv", ["answer_question.py", "--query", "synthetic"])
    assert answer_cli.main() == 0
    captured = capsys.readouterr()
    assert captured.out == "Synthetic safe answer\n"
    assert captured.err == ""
    instances = cast(list[object], getattr(pipeline_type, "instances"))
    assert getattr(instances[0], "diagnostic_flags") == [False]
    assert not list((tmp_path / "data/run-reports").glob("*.json"))
