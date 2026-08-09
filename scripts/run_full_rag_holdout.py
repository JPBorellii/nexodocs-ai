"""CLI for the direct, privacy-safe full-RAG holdout R03 runner."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import cast

from _project_bootstrap import bootstrap_project


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit single-attempt CLI contract."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", choices=("r03",), required=True)
    parser.add_argument("--snapshot-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--contract-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--report-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--qdrant-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--parent-attestation-sha256", help=argparse.SUPPRESS)
    parser.add_argument(
        "--snapshot-child-preflight-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _validate_contracts(project: Path, *, verify_git_provenance: bool = True) -> None:
    from validate_evaluation_oracle_corrections import validate_oracle_corrections
    from validate_full_rag_holdout_fixture import validate_fixture
    from validate_full_rag_system_freeze import validate_system_freeze

    from nexodocs_ai.rag.full_rag_holdout_r03 import HoldoutHarnessError

    try:
        validate_oracle_corrections(project)
        validate_fixture(project, "r03")
        validate_system_freeze(
            project,
            "r03",
            verify_git_provenance=verify_git_provenance,
        )
    except Exception as exc:
        raise HoldoutHarnessError("fixture_or_freeze_invalid") from exc


def _configuration(qdrant_path: Path | None = None):  # type annotation is local-imported below
    from nexodocs_ai.rag.config import load_rag_config
    from nexodocs_ai.rag.full_rag_holdout_r03 import RuntimeConfiguration
    from nexodocs_ai.retrieval.config import load_config

    retrieval = load_config()
    if qdrant_path is not None:
        retrieval = replace(retrieval, qdrant_path=str(qdrant_path.resolve()))
    return RuntimeConfiguration(retrieval, load_rag_config())


def _child_main(
    contract_root: Path,
    report_root: Path,
    qdrant_path: Path,
    parent_attestation_sha256: str,
    *,
    preflight_only: bool = False,
) -> int:
    """Execute entirely from captured SUT and harness bytes in one child process."""
    from nexodocs_ai.rag.full_rag_holdout_r03 import preflight_r03

    configuration = _configuration(qdrant_path)

    def child_contract_validator(project: Path) -> None:
        _validate_contracts(project, verify_git_provenance=False)

    if preflight_only:
        preflight_r03(
            contract_root,
            report_root,
            configuration,
            contract_validator=child_contract_validator,
            parent_attestation_sha256=parent_attestation_sha256,
        )
        return 0

    from nexodocs_ai.rag.answer_provider import create_answer_provider
    from nexodocs_ai.rag.full_rag_holdout_r03 import (
        HoldoutHarnessError,
        IndexStore,
        execute_r03,
    )
    from nexodocs_ai.rag.full_rag_holdout_r03_integrity import VectorRecord
    from nexodocs_ai.rag.pipeline import RagPipeline
    from nexodocs_ai.retrieval.embeddings import create_embedding_provider
    from nexodocs_ai.retrieval.qdrant_store import QdrantStore, create_client
    from nexodocs_ai.retrieval.retriever import Retriever

    class R03QdrantStore(QdrantStore):
        """Read-only vector scroll added by the harness without changing the SUT."""

        def all_vector_records(self) -> list[VectorRecord]:
            records: list[VectorRecord] = []
            offset = None
            while True:
                page, offset = self.client.scroll(
                    self.collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                for record in page:
                    vector = record.vector
                    if not isinstance(vector, list) or not all(
                        isinstance(component, int | float) for component in vector
                    ):
                        raise HoldoutHarnessError("vector_shape_invalid")
                    numeric_vector = cast(list[float], vector)
                    records.append(
                        VectorRecord(
                            str(record.id),
                            dict(record.payload or {}),
                            tuple(float(component) for component in numeric_vector),
                        )
                    )
                if offset is None:
                    return records

    stores: list[R03QdrantStore] = []
    closed_store_ids: set[int] = set()

    def create_store() -> R03QdrantStore:
        if len(stores) >= 2:
            raise HoldoutHarnessError("store_recreation_forbidden")
        retrieval = configuration.retrieval
        store = R03QdrantStore(
            create_client(retrieval),
            retrieval.collection_name,
            retrieval.embedding_dimensions,
        )
        stores.append(store)
        return store

    def close_store(bound_store: IndexStore) -> None:
        if not isinstance(bound_store, R03QdrantStore) or bound_store not in stores:
            raise HoldoutHarnessError("same_store_invariant_failed")
        identity = id(bound_store)
        if identity in closed_store_ids:
            raise HoldoutHarnessError("same_store_invariant_failed")
        bound_store.client.close()
        closed_store_ids.add(identity)

    def create_pipeline(bound_store: IndexStore) -> RagPipeline:
        if (
            len(stores) != 1
            or bound_store is not stores[0]
            or not isinstance(bound_store, R03QdrantStore)
        ):
            raise HoldoutHarnessError("same_store_invariant_failed")
        retrieval = configuration.retrieval
        embedding = create_embedding_provider(retrieval)
        if embedding.dimensions != retrieval.embedding_dimensions:
            raise HoldoutHarnessError("runtime_configuration_drift")
        retriever = Retriever(
            embedding,
            bound_store,
            retrieval.top_k,
            retrieval.max_top_k,
            retrieval.max_per_document,
        )
        if retriever.store is not bound_store:
            raise HoldoutHarnessError("same_store_invariant_failed")
        return RagPipeline(
            retriever,
            create_answer_provider(configuration.answer),
            configuration.answer,
        )

    try:
        execute_r03(
            contract_root,
            report_root,
            configuration,
            create_pipeline,
            contract_validator=child_contract_validator,
            store_factory=create_store,
            store_closer=close_store,
            parent_attestation_sha256=parent_attestation_sha256,
        )
    finally:
        for opened_store in reversed(stores):
            if id(opened_store) not in closed_store_ids:
                opened_store.client.close()
                closed_store_ids.add(id(opened_store))
    return 0


def _ordered_outputs() -> tuple[Path, ...]:
    from nexodocs_ai.rag.full_rag_holdout_r03 import CASE_ORDER, SUMMARY, usage_report_path

    return (*(usage_report_path(case_id) for case_id in CASE_ORDER), SUMMARY)


def _capture_staged(staging: Path) -> dict[Path, bytes]:
    """Capture every staged artifact once; these bytes become the publication set."""
    captured: dict[Path, bytes] = {}
    for relative in _ordered_outputs():
        source = (staging / relative).resolve()
        try:
            source.relative_to(staging.resolve())
            if source.is_symlink() or not source.is_file():
                raise RuntimeError("staged_result_invalid")
            captured[relative] = source.read_bytes()
        except OSError as exc:
            raise RuntimeError("staged_result_unavailable") from exc
    return captured


def _publish_validated(root: Path, artifacts: dict[Path, bytes]) -> None:
    from nexodocs_ai.rag.full_rag_holdout_r03 import publish_validated_outputs

    publish_validated_outputs(root, artifacts)


def _launcher_main(root: Path) -> int:
    from nexodocs_ai.rag.full_rag_holdout_r03 import (
        SUMMARY,
        HoldoutHarnessError,
        acquire_run_reservation,
        materialize_execution_snapshot,
        preflight_r03,
        usage_report_path,
        verify_preflight_integrity,
    )
    from nexodocs_ai.rag.full_rag_holdout_r03_integrity import child_process_failure_code
    from nexodocs_ai.rag.full_rag_holdout_r03_result import validate_result_bytes

    with acquire_run_reservation(root):
        configuration = _configuration()
        snapshot = preflight_r03(
            root,
            root,
            configuration,
            contract_validator=_validate_contracts,
        )
        captured_contract_contents = {
            item.relative_path: item.content for item in snapshot.integrity_files
        }
        report_parent = root / "data" / "run-reports"
        report_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".r03-execution-", dir=report_parent
        ) as execution_name:
            with tempfile.TemporaryDirectory(
                prefix=".r03-staging-", dir=report_parent
            ) as staging_name:
                execution_root = Path(execution_name)
                staging_root = Path(staging_name)
                child_snapshot = materialize_execution_snapshot(snapshot, execution_root)
                qdrant_path = (root / configuration.retrieval.qdrant_path).resolve()
                environment = dict(os.environ)
                environment["PYTHONPATH"] = str(execution_root / "src")
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(execution_root / "scripts" / "run_full_rag_holdout.py"),
                        "--attempt",
                        "r03",
                        "--snapshot-child",
                        "--contract-root",
                        str(execution_root),
                        "--report-root",
                        str(staging_root),
                        "--qdrant-path",
                        str(qdrant_path),
                        "--parent-attestation-sha256",
                        snapshot.parent_provenance_attestation.sha256,
                    ],
                    cwd=execution_root,
                    env=environment,
                    check=False,
                    capture_output=True,
                )
                if completed.returncode != 0:
                    raise HoldoutHarnessError(
                        child_process_failure_code(completed.stdout, completed.stderr)
                    )
                if completed.stdout or completed.stderr:
                    raise HoldoutHarnessError("snapshot_child_failed")
                verify_preflight_integrity(execution_root, child_snapshot)
                artifacts = _capture_staged(staging_root)
                usage_contents = {
                    case["case_id"]: artifacts[usage_report_path(case["case_id"])]
                    for case in snapshot.cases
                }
                validate_result_bytes(
                    execution_root,
                    artifacts[SUMMARY],
                    usage_contents,
                    captured_contract_contents=captured_contract_contents,
                )
                verify_preflight_integrity(execution_root, child_snapshot)
                _publish_validated(root, artifacts)
    return 0


def main() -> int:
    """Execute the explicitly selected real R03 attempt."""
    arguments = build_parser().parse_args()
    try:
        if arguments.snapshot_child:
            if (
                arguments.contract_root is None
                or arguments.report_root is None
                or arguments.qdrant_path is None
                or arguments.parent_attestation_sha256 is None
            ):
                return 2
            return _child_main(
                arguments.contract_root.resolve(),
                arguments.report_root.resolve(),
                arguments.qdrant_path.resolve(),
                arguments.parent_attestation_sha256,
                preflight_only=arguments.snapshot_child_preflight_only,
            )
        root = bootstrap_project()
        return _launcher_main(root)
    except Exception as exc:
        code = getattr(exc, "code", "runtime_configuration_invalid")
        from nexodocs_ai.rag.full_rag_holdout_r03_integrity import encode_child_error_envelope

        sys.stdout.buffer.write(encode_child_error_envelope(str(code)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
