# Full RAG holdout R03 offline preparation

## Scope

This preparation does not execute R03 and does not create operational evidence. The system under
test is the grounding boundary at commit `040082569a90bf318ee52f3bf622a9bd01427928`.
R01, R02, D03, D04, and the evaluation-oracle correction remain immutable historical evidence.

The prepared contracts are:

- `evals/rag/full-rag-holdout-r03-cases.json`;
- `evals/rag/full-rag-holdout-r03-cases.schema.json`;
- `evals/rag/full-rag-holdout-r03-system-freeze.json`;
- `evals/rag/full-rag-holdout-r03-system-freeze.schema.json`.
- `evals/rag/full-rag-holdout-r03-vector-fingerprint-v1.json` and its closed schema;
- `evals/rag/full-rag-holdout-r03-system-runtime-manifest-v1.json`;
- `evals/rag/full-rag-holdout-r03-evaluation-harness-manifest-v1.json`;
- `evals/rag/full-rag-holdout-r03-result.schema.json`.

R03 preserves all 12 R02 case IDs, kinds, queries, document expectations, safety outcomes, and
preflight behavior. The only case delta is the official P05 fact correction from
`evaluation-oracle-corrections-v1`: `nexo_integral_not_confirmed_for_north_unit`.

## Controlled pre-execution re-freeze

`PRE_EXECUTION_CONTROLLED_REFREEZE=true`. R03 has not been executed, so its preparation freeze was
re-issued to close vector-content, runtime-provenance, harness-provenance, and mutable-input
snapshot gaps. This is not a historical rewrite. The predecessor freeze SHA-256
recorded by each freeze is always its immediate predecessor. The complete append-only lineage is:

1. initial pre-execution freeze
   `5d3a9f9aad1014e9434eac56bb3c46f4d1fe2a26afb7bc1c1efb5bf2f93d2157`;
2. first controlled re-freeze
   `62e9d978278635c51b2aa693b17b97d351c2802f27f0f183c0a05e4bf0c110c2`;
3. pre-remediation controlled freeze
   `7bfd9126203e1d55a5824a9832864017b9df0ffefc8a77b38447c0ba40c2d211`;
4. current controlled freeze
   `eec4b2bec79d00ef42ffd61d81568b468f17c48f4c32d2ea7e7c2c7456dffb08`.

The current freeze records `7bfd9126...d211` as its immediate predecessor. R01, R02, R02
adjudication, D02, D03, D04, threshold policy, index plan, index manifest, and the R03 fixture
remain unchanged.

The current controlled freeze SHA-256 is
`eec4b2bec79d00ef42ffd61d81568b468f17c48f4c32d2ea7e7c2c7456dffb08`. It binds:

- system runtime manifest artifact SHA-256
  `78f89804fb369933585ab8d9d8a046daef640843619d08b8c322c9f663018663`, whose canonical
  path-and-file-digest aggregate is
  `69f8a807069cdb617e5b3050f33eaaac89824110619988e9a1452e33513ba3e8`;
- evaluation harness manifest artifact SHA-256
  `d16652cfbd88837b4358cf8d26b1a0b4f7565a430337f5b7bc12bcb27bad554c`, whose canonical aggregate
  is `8d6ab37ba176311c3a39503675bf9c930ace3e70234cd23c43671185d7fc0776`;
- vector fingerprint artifact SHA-256
  `acffe9de7beeb5ae6d4f4fecda5864d4ef49515c5cede3c4e633ad9e57396834`, semantic version `1.0.0`,
and vector aggregate `e22edc5db73d3b99f7a16ef7ead3f7c70df6a449fe5da03569af73fed3d8fa7d`.

Dependency-environment provenance is part of the system runtime manifest and is also explicit in
the freeze and result. The single publication authority is the immutable execution snapshot
captured by preflight: `pyproject.toml` is bound to
`0ab89ae7bf22d84727cc12960cf5e495d8c08433002dfdbb53993fd5a58fc9ae` and `uv.lock` to
`bfb0921aedd880bcdac33f1d6838107525ed3fc73d4e5c6db2e0afff6a629f44`. At capture time both byte
sets must equal their paths at the frozen system commit. Execution requires Python `==3.14.*`, the
`UV_RUN_RECURSION_DEPTH` marker produced by `uv run --locked`, and exact installed versions
`jsonschema==4.26.0`, `openai==2.52.0`, and `qdrant-client==1.18.0`, all attested in the result.
After materialization, child execution and parent validation read only the snapshot authority;
later changes to the original working paths neither alter nor invalidate that captured run.

Every vector is validated as exactly 1536 finite numeric components, converted explicitly to
IEEE-754 float32, serialized little-endian, and hashed with SHA-256. The persisted artifact contains
only point identity, chunk identity, payload digest, dimensions, and vector digest; it contains no
vector plaintext and no physical Qdrant path.

## Offline preflight

Run from the repository root in the real PowerShell session. These commands do not require a key,
Qdrant, or network access:

```powershell
uv run --locked python scripts/validate_evaluation_oracle_corrections.py
uv run --locked python scripts/validate_full_rag_holdout_fixture.py --attempt r03
uv run --locked python scripts/validate_full_rag_system_freeze.py --attempt r03
```

Stop if any command returns a nonzero exit code. Before a real execution, also confirm that no
`phase-6a-rag-holdout-r03-*.json` destination exists.

## Reserved operational artifact names

The 12 privacy-safe usage reports are reserved under `data/run-reports/`:

```text
phase-6a-rag-holdout-r03-p01.json
phase-6a-rag-holdout-r03-p02.json
phase-6a-rag-holdout-r03-p03.json
phase-6a-rag-holdout-r03-p04.json
phase-6a-rag-holdout-r03-p05.json
phase-6a-rag-holdout-r03-p06.json
phase-6a-rag-holdout-r03-n01.json
phase-6a-rag-holdout-r03-n02.json
phase-6a-rag-holdout-r03-n03.json
phase-6a-rag-holdout-r03-n04.json
phase-6a-rag-holdout-r03-n05.json
phase-6a-rag-holdout-r03-n06.json
```

The reserved consolidated result is
`data/run-reports/phase-6a-rag-holdout-r03-summary.json`. An adjudication artifact is not reserved
at preparation time; it is applicable only if a completed immutable summary later requires formal
adjudication.

None of these operational files is created by this preparation.

The launcher atomically creates `data/run-reports/.phase-6a-rag-holdout-r03.lock` before any Qdrant
or provider construction and holds it through complete publication or abort cleanup. Another R03
harness instance on the same report root fails closed before those factories run. A hard power loss
may leave a stale reservation; it is never auto-recovered. An operator may remove it only after
proving that no R03 execution is active and that no canonical run has been published. This protocol
coordinates instances of this harness; it does not claim protection from an arbitrary malicious
process that ignores the reservation.

## Future real execution

First configure the exact non-sensitive environment and masked key procedure in
`docs/runbooks/phase-6a-real-local-execution.md`. Do not change the values frozen by R03 and do not
use any overwrite option. After the offline preflight passes, run the dedicated harness once:

```powershell
uv run --locked python scripts/run_full_rag_holdout.py --attempt r03
uv run --locked python scripts/validate_full_rag_holdout_r03_result.py
```

The launcher first captures exact bytes for the fixture, parsed cases, freeze, vector fingerprint,
both manifests, prompts, generated-answer schema, privacy-safe usage schema, result schema,
historical contracts, threshold/index bindings, destination plan, and validated configuration. It
materializes only those captured bytes into an isolated execution root. The SUT is imported from
that root, so its existing prompt and generated-schema loaders consume the captured copy without
changing the bytes or semantics of system commit `040082569a90bf318ee52f3bf622a9bd01427928`.
The result writer validates with the preflight-captured result-schema bytes. After the child exits,
the parent captures each sanitized staged artifact exactly once, validates those captured bytes,
performs the final integrity check against the materialized execution root, and publishes exactly
that immutable byte set. It never rereads the original mutable contract root or staging to
determine canonical content.

The child then opens only the configured local Qdrant store and scrolls all 44 records with payloads
and vectors. It compares exact point IDs, chunk/document identities, text and canonical payload
hashes, vector dimensions, per-point vector hashes, and the aggregate vector fingerprint. The same
store object is passed to the Retriever and used for the second full binding immediately before
summary construction. Embedding and answer providers are constructed only after the first binding
succeeds; a machine-specific Qdrant path is not part of the semantic binding. Any vector mutation,
missing/extra point, dimension mismatch, runtime/harness drift, or historical-contract drift fails
closed with no summary.

The runner calls `RagPipeline.answer_with_usage` directly. Each `RagResponse` exists only in memory,
is evaluated immediately, and is projected to closed booleans and safe status codes. Required-fact
coverage reads only the assistant-authored `answer`; citations and supporting excerpts are checked
separately and cannot supply a missing fact. Unsupported cases pass only through the official
`no_evidence` status and expected reason-code contract. The 12
per-case files remain canonical privacy-safe usage reports; only the final summary contains the
sanitized case evaluations. The summary is published exclusively and atomically after all cases
complete, and binds the fixture, freeze, threshold policy, index manifest, result schema, both
provenance manifests, non-vector index content, vector artifact and aggregate, and every usage
report by SHA-256. The system manifest conservatively covers the dependency inputs, RAG, retrieval
and observability runtime plus prompts and dynamic schemas; every current runtime byte is also
compared with the same path at the frozen system commit. The separate harness manifest covers preflight, vector checking,
execution, semantic evaluation, result validation, schemas, and invoked contract validators. The
offline result validator independently enforces the closed case/status/reason/boolean state machine
and recalculates numerators, denominators, rates, failed cases, and decision.

Before constructing any provider or Qdrant client, the runner validates the fixture, freeze,
historical oracle correction, runtime configuration, frozen system state, case order, schemas, and
all 13 reserved destinations. It constructs the local Qdrant client only for the content-binding
step described above; no OpenAI provider can be constructed before that binding. Any collision,
contract drift, or index-content mismatch aborts with zero provider calls. A technical
failure stops the run with a sanitized code and never publishes a quality summary. Existing usage
reports are never overwritten. Canonical usage reports are committed from parent-owned temporary
files and the summary is committed last. A publication failure prevents summary publication and
rolls back only files whose hashes still prove they were created by that attempt; a changed or
unprovable path is never deleted. Temporary-file cleanup after a successful exclusive commit is
best effort and cannot turn canonical success into a false failure. P05 is evaluated only against
`nexo_integral_not_confirmed_for_north_unit`; P02 and P04 receive no case-specific exception and use
the same pipeline grounding validator as every other case.

The offline directed tests use an output root under pytest's temporary directory and injected,
deterministic pipeline results. They never create canonical R03 artifacts or contact OpenAI/Qdrant.
