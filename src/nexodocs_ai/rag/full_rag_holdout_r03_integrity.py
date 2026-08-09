"""Closed provenance and vector-integrity primitives for the R03 harness."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import struct
import subprocess
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

VECTOR_ENCODING = "ieee754-float32-little-endian-v1"
VECTOR_DIMENSIONS = 1536
PARENT_ATTESTATION_ARTIFACT_ID = "full-rag-holdout-r03-parent-attestation-v1"
MAX_CHILD_ERROR_ENVELOPE_BYTES = 256
CHILD_SAFE_ERROR_CODES = frozenset(
    {
        "answer_provider_failed",
        "case_evaluation_failed",
        "case_execution_failed",
        "case_order_invalid",
        "contract_integrity_changed",
        "destination_collision",
        "destination_invalid",
        "exclusive_result_publication_failed",
        "execution_snapshot_conflict",
        "execution_snapshot_destination_not_empty",
        "execution_snapshot_path_invalid",
        "fixture_invalid",
        "fixture_or_freeze_invalid",
        "freeze_binding_invalid",
        "index_binding_changed",
        "index_binding_failed",
        "index_binding_strategy_invalid",
        "index_binding_unavailable",
        "index_contract_invalid",
        "index_manifest_integrity_invalid",
        "integrity_source_unavailable",
        "manifest_file_changed",
        "manifest_file_unavailable",
        "parent_attestation_binding_invalid",
        "parent_attestation_hash_invalid",
        "parent_attestation_invalid",
        "parent_attestation_missing",
        "parent_git_root_invalid",
        "pipeline_initialization_failed",
        "preflight_contract_unavailable",
        "preflight_snapshot_incomplete",
        "provenance_integrity_changed",
        "provenance_manifest_unavailable",
        "provenance_manifest_aggregate_invalid",
        "provenance_manifest_invalid",
        "reservation_cleanup_failed",
        "reservation_exists",
        "reservation_failed",
        "reservation_path_invalid",
        "result_schema_invalid",
        "runtime_configuration_drift",
        "runtime_configuration_invalid",
        "runtime_dependency_unavailable",
        "runtime_dependency_version_mismatch",
        "runtime_python_version_mismatch",
        "safe_outcome_invalid",
        "same_store_invariant_failed",
        "snapshot_child_failed",
        "store_recreation_forbidden",
        "summary_schema_validation_failed",
        "summary_write_failed",
        "system_commit_mismatch",
        "threshold_policy_integrity_invalid",
        "unknown_fact_code",
        "unknown_safe_outcome",
        "usage_report_write_failed",
        "uv_lock_critical_package_invalid",
        "uv_lock_invalid",
        "uv_lock_python_contract_invalid",
        "uv_locked_launcher_required",
        "validated_artifact_set_invalid",
        "vector_binding_unavailable",
        "vector_component_invalid",
        "vector_content_mismatch",
        "vector_dimensions_invalid",
        "vector_fingerprint_aggregate_invalid",
        "vector_fingerprint_invalid",
        "vector_index_plan_mismatch",
        "vector_payload_invalid",
        "vector_shape_invalid",
    }
)


class R03IntegrityError(RuntimeError):
    """Raised when closed provenance or vector content does not match."""

    def __init__(self, code: str) -> None:
        super().__init__("R03 integrity validation failed")
        self.code = code


@dataclass(frozen=True)
class FileBinding:
    """One path-and-digest entry in a deterministic manifest."""

    path: Path
    sha256: str

    def as_json(self) -> dict[str, str]:
        """Return the canonical JSON projection."""
        return {"path": self.path.as_posix(), "sha256": self.sha256}


@dataclass(frozen=True)
class FileContent:
    """Immutable bytes captured for one manifest-bound file."""

    path: Path
    sha256: str
    content: bytes = field(repr=False)


@dataclass(frozen=True)
class ProvenanceManifest:
    """Closed manifest parsed from its exact artifact bytes."""

    artifact_id: str
    semantic_version: str
    aggregate_sha256: str
    files: tuple[FileBinding, ...]


@dataclass(frozen=True)
class VectorPointBinding:
    """Privacy-safe identity and vector digest for one Qdrant point."""

    point_id: str
    chunk_id: str
    payload_sha256: str
    vector_dimensions: int
    vector_sha256: str

    def as_json(self) -> dict[str, object]:
        """Return the canonical aggregate entry."""
        return {
            "chunk_id": self.chunk_id,
            "payload_sha256": self.payload_sha256,
            "point_id": self.point_id,
            "vector_dimensions": self.vector_dimensions,
            "vector_sha256": self.vector_sha256,
        }


@dataclass(frozen=True)
class VectorFingerprint:
    """Frozen vector content expected from the canonical collection."""

    artifact_id: str
    collection_name: str
    distance: str
    vector_dimensions: int
    vector_encoding: str
    aggregate_sha256: str
    points: tuple[VectorPointBinding, ...]


@dataclass(frozen=True)
class VectorRecord:
    """Ephemeral vector-bearing record; vector values are never persisted."""

    point_id: str
    payload: Mapping[str, object]
    vector: tuple[float, ...] = field(repr=False)


class VectorStore(Protocol):
    """Read-only same-store boundary required by the R03 harness."""

    def validate_collection(self) -> None: ...

    def all_vector_records(self) -> list[VectorRecord]: ...

    def count(self) -> int: ...


def canonical_json_bytes(value: object) -> bytes:
    """Encode JSON deterministically without locale or whitespace variance."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 digest for exact bytes."""
    return hashlib.sha256(content).hexdigest()


def manifest_aggregate(files: Sequence[FileBinding]) -> str:
    """Hash the ordered canonical path-and-digest manifest entries."""
    ordered = sorted(files, key=lambda item: item.path.as_posix())
    return sha256_bytes(canonical_json_bytes([item.as_json() for item in ordered]))


def _closed_bindings(files: Sequence[FileBinding]) -> tuple[FileBinding, ...]:
    ordered = tuple(sorted(files, key=lambda item: item.path.as_posix()))
    if len({item.path for item in ordered}) != len(ordered):
        raise R03IntegrityError("parent_attestation_binding_invalid")
    for item in ordered:
        path = item.path.as_posix()
        if (
            item.path.is_absolute()
            or path in {"", "."}
            or "\\" in path
            or ".." in item.path.parts
            or re.fullmatch(r"[a-f0-9]{64}", item.sha256) is None
        ):
            raise R03IntegrityError("parent_attestation_binding_invalid")
    return ordered


def build_parent_provenance_attestation(
    system_commit: str,
    system_manifest_path: Path,
    system_manifest_sha256: str,
    system_manifest_aggregate_sha256: str,
    system_files: Sequence[FileBinding],
    snapshot_files: Sequence[FileBinding],
) -> bytes:
    """Build the canonical Git-authoritative proof consumed by one snapshot child."""
    if (
        re.fullmatch(r"[a-f0-9]{40}", system_commit) is None
        or re.fullmatch(r"[a-f0-9]{64}", system_manifest_sha256) is None
        or re.fullmatch(r"[a-f0-9]{64}", system_manifest_aggregate_sha256) is None
        or system_manifest_path.is_absolute()
        or ".." in system_manifest_path.parts
    ):
        raise R03IntegrityError("parent_attestation_binding_invalid")
    closed_system = _closed_bindings(system_files)
    closed_snapshot = _closed_bindings(snapshot_files)
    if manifest_aggregate(closed_system) != system_manifest_aggregate_sha256:
        raise R03IntegrityError("parent_attestation_binding_invalid")
    value = {
        "artifact_id": PARENT_ATTESTATION_ARTIFACT_ID,
        "schema_version": "1.0",
        "snapshot": {
            "aggregate_sha256": manifest_aggregate(closed_snapshot),
            "files": [item.as_json() for item in closed_snapshot],
        },
        "system_commit": system_commit,
        "system_runtime_manifest": {
            "aggregate_sha256": system_manifest_aggregate_sha256,
            "files": [item.as_json() for item in closed_system],
            "path": system_manifest_path.as_posix(),
            "sha256": system_manifest_sha256,
        },
    }
    return canonical_json_bytes(value)


def validate_parent_provenance_attestation(
    content: bytes,
    expected_sha256: str,
    *,
    system_commit: str,
    system_manifest_path: Path,
    system_manifest_sha256: str,
    system_manifest_aggregate_sha256: str,
    system_files: Sequence[FileBinding],
    snapshot_files: Sequence[FileBinding],
) -> None:
    """Validate exact canonical attestation bytes without consulting Git."""
    if (
        re.fullmatch(r"[a-f0-9]{64}", expected_sha256) is None
        or sha256_bytes(content) != expected_sha256
    ):
        raise R03IntegrityError("parent_attestation_hash_invalid")
    expected = build_parent_provenance_attestation(
        system_commit,
        system_manifest_path,
        system_manifest_sha256,
        system_manifest_aggregate_sha256,
        system_files,
        snapshot_files,
    )
    if content != expected:
        raise R03IntegrityError("parent_attestation_binding_invalid")


def encode_child_error_envelope(code: str) -> bytes:
    """Encode one finite, canonical and privacy-safe child failure envelope."""
    safe_code = code if code in CHILD_SAFE_ERROR_CODES else "runtime_configuration_invalid"
    content = canonical_json_bytes({"safe_error_code": safe_code, "status": "holdout_failed"})
    if len(content) > MAX_CHILD_ERROR_ENVELOPE_BYTES:  # pragma: no cover - constant invariant
        raise AssertionError("child error envelope exceeds its closed bound")
    return content


def parse_child_error_envelope(stdout: bytes, stderr: bytes) -> str | None:
    """Accept only the exact finite child failure channel; reject all other output."""
    if stderr or not stdout or len(stdout) > MAX_CHILD_ERROR_ENVELOPE_BYTES:
        return None
    try:
        decoded = stdout.decode("utf-8")
        value: object = json.loads(decoded)
    except UnicodeDecodeError, json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    envelope = cast(Mapping[str, object], value)
    if set(envelope) != {"safe_error_code", "status"}:
        return None
    code = envelope.get("safe_error_code")
    if envelope.get("status") != "holdout_failed" or not isinstance(code, str):
        return None
    if code not in CHILD_SAFE_ERROR_CODES or stdout != encode_child_error_envelope(code):
        return None
    return code


def child_process_failure_code(stdout: bytes, stderr: bytes) -> str:
    """Preserve a valid child code or return the fixed fail-closed fallback."""
    return parse_child_error_envelope(stdout, stderr) or "snapshot_child_failed"


def capture_bound_files(root: Path, manifest: ProvenanceManifest) -> tuple[FileContent, ...]:
    """Capture and verify every file named by a closed provenance manifest."""
    captured: list[FileContent] = []
    for binding in manifest.files:
        candidate = (root / binding.path).resolve()
        try:
            candidate.relative_to(root.resolve())
            content = candidate.read_bytes()
        except (ValueError, OSError) as exc:
            raise R03IntegrityError("manifest_file_unavailable") from exc
        digest = sha256_bytes(content)
        if digest != binding.sha256:
            raise R03IntegrityError("manifest_file_changed")
        captured.append(FileContent(binding.path, digest, content))
    return tuple(captured)


def _json_object(content: bytes, code: str) -> Mapping[str, object]:
    try:
        value: object = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R03IntegrityError(code) from exc
    if not isinstance(value, dict):
        raise R03IntegrityError(code)
    return cast(Mapping[str, object], value)


def _required_string(value: Mapping[str, object], key: str, code: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise R03IntegrityError(code)
    return item


def load_provenance_manifest(content: bytes, expected_artifact_id: str) -> ProvenanceManifest:
    """Parse a closed manifest and independently verify its canonical aggregate."""
    value = _json_object(content, "provenance_manifest_invalid")
    if set(value) != {
        "schema_version",
        "artifact_id",
        "semantic_version",
        "aggregate_sha256",
        "files",
    }:
        raise R03IntegrityError("provenance_manifest_invalid")
    if value.get("schema_version") != "1.0" or value.get("artifact_id") != expected_artifact_id:
        raise R03IntegrityError("provenance_manifest_invalid")
    raw_files = value.get("files")
    if not isinstance(raw_files, list):
        raise R03IntegrityError("provenance_manifest_invalid")
    files: list[FileBinding] = []
    for raw in cast(list[object], raw_files):
        if not isinstance(raw, dict):
            raise R03IntegrityError("provenance_manifest_invalid")
        mapping = cast(Mapping[str, object], raw)
        if set(mapping) != {"path", "sha256"}:
            raise R03IntegrityError("provenance_manifest_invalid")
        path = Path(_required_string(mapping, "path", "provenance_manifest_invalid"))
        digest = _required_string(mapping, "sha256", "provenance_manifest_invalid")
        if path.is_absolute() or "\\" in path.as_posix() or len(digest) != 64:
            raise R03IntegrityError("provenance_manifest_invalid")
        files.append(FileBinding(path, digest))
    ordered = tuple(sorted(files, key=lambda item: item.path.as_posix()))
    if tuple(files) != ordered or len({item.path for item in ordered}) != len(ordered):
        raise R03IntegrityError("provenance_manifest_invalid")
    aggregate = _required_string(value, "aggregate_sha256", "provenance_manifest_invalid")
    if aggregate != manifest_aggregate(ordered):
        raise R03IntegrityError("provenance_manifest_aggregate_invalid")
    return ProvenanceManifest(
        expected_artifact_id,
        _required_string(value, "semantic_version", "provenance_manifest_invalid"),
        aggregate,
        ordered,
    )


def verify_system_commit(
    root: Path,
    system_commit: str,
    files: Sequence[FileContent],
) -> None:
    """Prove current runtime bytes equal the exact bytes in the frozen SUT commit."""
    completed = subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            system_commit,
            "--",
            *(item.path.as_posix() for item in files),
        ],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise R03IntegrityError("system_commit_mismatch")
    expected = {item.path.as_posix(): item.sha256 for item in files}
    observed: dict[str, str] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
            for member in archive.getmembers():
                if not member.isfile() or member.name not in expected:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise R03IntegrityError("system_commit_mismatch")
                observed[member.name] = sha256_bytes(extracted.read())
    except (tarfile.TarError, OSError) as exc:
        raise R03IntegrityError("system_commit_mismatch") from exc
    if observed != expected:
        raise R03IntegrityError("system_commit_mismatch")


def vector_sha256(vector: Sequence[float], dimensions: int = VECTOR_DIMENSIONS) -> str:
    """Hash exact IEEE-754 float32 components in fixed little-endian order."""
    if len(vector) != dimensions:
        raise R03IntegrityError("vector_dimensions_invalid")
    normalized: list[float] = []
    for component in vector:
        if isinstance(component, bool):
            raise R03IntegrityError("vector_component_invalid")
        numeric = float(component)
        if not math.isfinite(numeric):
            raise R03IntegrityError("vector_component_invalid")
        normalized.append(numeric)
    try:
        encoded = struct.pack(f"<{dimensions}f", *normalized)
    except (OverflowError, struct.error) as exc:
        raise R03IntegrityError("vector_component_invalid") from exc
    return sha256_bytes(encoded)


def vector_aggregate(points: Sequence[VectorPointBinding]) -> str:
    """Hash sorted privacy-safe point bindings without vector plaintext."""
    ordered = sorted(points, key=lambda item: item.point_id)
    return sha256_bytes(canonical_json_bytes([item.as_json() for item in ordered]))


def load_vector_fingerprint(content: bytes) -> VectorFingerprint:
    """Parse and semantically validate the closed vector fingerprint artifact."""
    value = _json_object(content, "vector_fingerprint_invalid")
    expected_keys = {
        "schema_version",
        "artifact_id",
        "collection_name",
        "distance",
        "vector_dimensions",
        "vector_encoding",
        "point_count",
        "points",
        "aggregate_sha256",
    }
    if set(value) != expected_keys:
        raise R03IntegrityError("vector_fingerprint_invalid")
    if (
        value.get("schema_version") != "1.0"
        or value.get("artifact_id") != "full-rag-holdout-r03-vector-fingerprint-v1"
        or value.get("collection_name") != "nexodocs_chunks_v1"
        or value.get("distance") != "Cosine"
        or value.get("vector_dimensions") != VECTOR_DIMENSIONS
        or value.get("vector_encoding") != VECTOR_ENCODING
        or value.get("point_count") != 44
    ):
        raise R03IntegrityError("vector_fingerprint_invalid")
    raw_points = value.get("points")
    if not isinstance(raw_points, list):
        raise R03IntegrityError("vector_fingerprint_invalid")
    points: list[VectorPointBinding] = []
    for raw in cast(list[object], raw_points):
        if not isinstance(raw, dict):
            raise R03IntegrityError("vector_fingerprint_invalid")
        mapping = cast(Mapping[str, object], raw)
        if set(mapping) != {
            "chunk_id",
            "payload_sha256",
            "point_id",
            "vector_dimensions",
            "vector_sha256",
        }:
            raise R03IntegrityError("vector_fingerprint_invalid")
        if mapping.get("vector_dimensions") != VECTOR_DIMENSIONS:
            raise R03IntegrityError("vector_fingerprint_invalid")
        points.append(
            VectorPointBinding(
                _required_string(mapping, "point_id", "vector_fingerprint_invalid"),
                _required_string(mapping, "chunk_id", "vector_fingerprint_invalid"),
                _required_string(mapping, "payload_sha256", "vector_fingerprint_invalid"),
                VECTOR_DIMENSIONS,
                _required_string(mapping, "vector_sha256", "vector_fingerprint_invalid"),
            )
        )
    ordered = tuple(sorted(points, key=lambda item: item.point_id))
    aggregate = _required_string(value, "aggregate_sha256", "vector_fingerprint_invalid")
    if (
        tuple(points) != ordered
        or len(ordered) != 44
        or len({item.point_id for item in ordered}) != 44
        or aggregate != vector_aggregate(ordered)
    ):
        raise R03IntegrityError("vector_fingerprint_aggregate_invalid")
    return VectorFingerprint(
        "full-rag-holdout-r03-vector-fingerprint-v1",
        "nexodocs_chunks_v1",
        "Cosine",
        VECTOR_DIMENSIONS,
        VECTOR_ENCODING,
        aggregate,
        ordered,
    )


def _payload_sha256(payload: Mapping[str, object]) -> str:
    try:
        content = (
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R03IntegrityError("vector_payload_invalid") from exc
    return sha256_bytes(content)


def validate_vector_binding(store: VectorStore, expected: VectorFingerprint) -> str:
    """Validate exact IDs, payload identities and vectors from one opened store."""
    try:
        store.validate_collection()
        records = store.all_vector_records()
        count = store.count()
    except R03IntegrityError:
        raise
    except Exception as exc:
        raise R03IntegrityError("vector_binding_unavailable") from exc
    actual: list[VectorPointBinding] = []
    for record in records:
        chunk_id = record.payload.get("chunk_id")
        if not isinstance(chunk_id, str):
            raise R03IntegrityError("vector_payload_invalid")
        actual.append(
            VectorPointBinding(
                record.point_id,
                chunk_id,
                _payload_sha256(record.payload),
                expected.vector_dimensions,
                vector_sha256(record.vector, expected.vector_dimensions),
            )
        )
    ordered = tuple(sorted(actual, key=lambda item: item.point_id))
    if (
        count != len(expected.points)
        or len(records) != count
        or len({item.point_id for item in ordered}) != count
        or ordered != expected.points
    ):
        raise R03IntegrityError("vector_content_mismatch")
    aggregate = vector_aggregate(ordered)
    if aggregate != expected.aggregate_sha256:
        raise R03IntegrityError("vector_content_mismatch")
    return aggregate
