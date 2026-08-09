# Full RAG holdout R03 offline preparation

## Scope

This preparation activity does not execute R03 or create new operational evidence. The system under
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

## R03 launcher attempt #1 technical incident

A first real launcher attempt occurred; it was not a completed R03 quality evaluation. The
available evidence does not establish an operational timestamp, so this record does not assert
one. The invoked command was:

```powershell
uv run --locked python scripts/run_full_rag_holdout.py --attempt r03
```

The launcher returned the safe visible error `snapshot_child_failed` with exit code `1`. The child
failed during snapshot preflight before any R03 case or external boundary was opened: cases
executed `0`, Qdrant opened `NO`, provider construction `NO`, and R03 OpenAI calls `0`. It produced
zero canonical reports, no summary, and no completed quality evaluation. Therefore this event is a
technical incident (`TECHNICAL_INCIDENT=YES`), not `FULL_RAG_HOLDOUT_FAILED`; no quality PASS/FAIL
was produced (`QUALITY_FAILURE=NO`).

The closed historical classification is: `R03_LAUNCHER_ATTEMPT_HAPPENED=YES`,
`TECHNICAL_INCIDENT=YES`, `QUALITY_EVALUATION_COMPLETED=NO`, `QUALITY_PASS_FAIL_PRODUCED=NO`,
`CASES_EXECUTED=0`, `CANONICAL_REPORTS=0`, and `SUMMARY_PRESENT=NO`.

Investigation identified the isolated snapshot child's Git provenance dependency as the cause. The
subsequent harness correction retained Git authority in the parent, created a closed provenance
attestation from the captured snapshot, and made the child validate that attestation and its local
bytes without Git. A later independent audit found and prompted closure of post-preflight semantic
prompt/schema rereads: the child now binds the authenticated prompt and generated-answer-schema
bytes in memory before store or provider construction. A later authorized launcher invocation
stopped with the sanitized code `vector_content_mismatch` and published no canonical summary or
result. Post-incident validation proved that the persisted 44-point collection and frozen vector
fingerprint remained unchanged. An isolated no-provider reproduction then established that
qdrant-client 1.18.0 local/Cosine search normalizes its loaded dense matrix in place, making a
second byte-exact check of the same searched instance a harness false positive.

## Controlled pre-execution re-freeze

`PRE_EXECUTION_CONTROLLED_REFREEZE=true`. No R03 quality evaluation has completed; launcher attempt
#1 executed zero cases and created no operational artifacts. The preparation freeze was re-issued
to close vector-content, runtime-provenance, harness-provenance, and mutable-input snapshot gaps.
This is not a historical rewrite. The predecessor freeze SHA-256 recorded by each freeze is always
its immediate predecessor. The complete append-only lineage is:

1. initial pre-execution freeze
   `5d3a9f9aad1014e9434eac56bb3c46f4d1fe2a26afb7bc1c1efb5bf2f93d2157`;
2. first controlled re-freeze
   `62e9d978278635c51b2aa693b17b97d351c2802f27f0f183c0a05e4bf0c110c2`;
3. pre-remediation controlled freeze
   `7bfd9126203e1d55a5824a9832864017b9df0ffefc8a77b38447c0ba40c2d211`;
4. pre-incident-fix controlled freeze
   `eec4b2bec79d00ef42ffd61d81568b468f17c48f4c32d2ea7e7c2c7456dffb08`;
5. pre-Ruff-microfix controlled freeze
   `55175faa5d25d236463afc60212507b2a07dcd0f38913394ef1ef3c7e3b25690`;
6. pre-audit-blockers-fix controlled freeze
   `22ee13ff9c024094eb273b874535324c611e54d1b1eb1dad14247a52b0161de8`;
7. pre-Ruff-format-microfix controlled freeze
   `aecd49481598b66f81009e8ce5625ef6a9a203f9e027c67ff50809a9eddf83f1`;
8. pre-Pyright-semantic-binding-microfix controlled freeze
   `dc72786935b375f67a5ffcad7da996d87da463b3c66607e6fc716ecdb26a58e0`;
9. pre-real-child-runtime-drift-fix controlled freeze
   `46dbb45185126f42e020153ab837a95759ae79be5f8b8b47f713438e07d702db`;
10. pre-local-Cosine-lifecycle-fix controlled freeze
    `33baa8a50b106542e8e5d0420234c56630d569be32b9af23aec8f51c04c6cb9f`;
11. pre-publication-rollback-audit-fix controlled freeze
    `d5fc4374defbe9d480e0a35b3cd06f50f5b51ab0d1f652fb5af36d3b16636aee`;
12. current controlled freeze
    `9cfb6281138e89bdf1d2442ca7422ad3af5c650997f5fa68a8a8e26a835d4d7c`.

The current freeze records `d5fc4374...636aee` as its immediate predecessor. R01, R02, R02
adjudication, D02, D03, D04, threshold policy, index plan, index manifest, and the R03 fixture
remain unchanged.

The current controlled freeze SHA-256 is
`9cfb6281138e89bdf1d2442ca7422ad3af5c650997f5fa68a8a8e26a835d4d7c`. It binds:

- system runtime manifest artifact SHA-256
  `78f89804fb369933585ab8d9d8a046daef640843619d08b8c322c9f663018663`, whose canonical
  path-and-file-digest aggregate is
  `69f8a807069cdb617e5b3050f33eaaac89824110619988e9a1452e33513ba3e8`;
- evaluation harness manifest artifact SHA-256
  `673e0b5c5f3e6069cc480f92daf9b1a85e4d09835a643ad9de3f206437236f27`, whose canonical aggregate
  is `bbb18db0e11b6d64dee7866b2aa7dae4a2d0476421d20da1886073bfbde2ffd3`;
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

The launcher first proves the frozen runtime bytes against Git from the real repository root, then
captures exact bytes for the fixture, parsed cases, freeze, vector fingerprint,
both manifests, prompts, generated-answer schema, privacy-safe usage schema, result schema,
historical contracts, threshold/index bindings, destination plan, and validated configuration. It
creates a canonical parent attestation, and materializes both the captured bytes and attestation
into an isolated execution root. The attestation binds the frozen system commit, runtime-manifest
artifact and aggregate, its ordered path/hash set, and the complete ordered snapshot path/hash set.
Its exact SHA-256 is passed out-of-band in the child arguments. The child imports the SUT from
that root without changing the code bytes or semantics of system commit
`040082569a90bf318ee52f3bf622a9bd01427928`.
The child validates the exact attestation and all local manifests and bytes without locating
`.git`, deriving a repository root from `__file__`, or executing Git. Qdrant, embedding-provider,
and answer-provider construction remains after this complete child preflight. Immediately after
preflight, a child-confined binding parses the captured prompt/schema bytes once, retains them as an
immutable semantic authority in memory, and binds only the SUT prompt and generated-answer-schema
loaders. Prompt rendering, OpenAI Structured Outputs, and returned-answer schema validation all use
that same authority; schema calls receive defensive copies. Later filesystem changes cannot affect
provider requests, while the final integrity check still rejects changed snapshot bytes.
The result writer validates with the preflight-captured result-schema bytes. After the child exits,
the parent captures each sanitized staged artifact exactly once, validates those captured bytes,
and supplies its in-memory preflight contract-byte map to result validation. The launcher does not
reread semantic contracts from the execution root for that decision. It then performs the final
integrity check against the materialized execution root and publishes exactly the immutable staged
byte set. It never rereads the original mutable contract root or staging to determine canonical
content.

The child then opens only the configured local Qdrant store and scrolls all 44 records with payloads
and vectors. It compares exact point IDs, chunk/document identities, text and canonical payload
hashes, vector dimensions, per-point vector hashes, and the aggregate vector fingerprint. That one
execution store object is passed to the Retriever and used for every case. After the last retrieval,
the child closes it, reopens the exact same canonical local Qdrant path with a fresh client, repeats
the full exact binding against newly loaded persisted bytes, and closes the audit client. No
retrieval occurs after the execution store is closed. Embedding and answer providers are constructed
only after the first binding succeeds; a machine-specific Qdrant path is not part of the semantic
binding. Any persisted vector or payload mutation, missing/extra point, dimension mismatch,
runtime/harness drift, reopen failure, or historical-contract drift fails closed with no summary.

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

## Finalized historical outcome

R03 completed as a technically valid quality evaluation and is now finalized, exposed historical
evidence. The immutable decision is `FULL_RAG_HOLDOUT_FAILED`: 6 of 12 cases passed, and the failed
case IDs are `HOLD-P02`, `HOLD-P04`, `HOLD-P05`, `HOLD-N01`, `HOLD-N02`, and `HOLD-N04`.

The tracked sanitized finalization record is
`evals/rag/full-rag-holdout-r03-finalization-v1.json`. It is a projection authenticated against the
unchanged operational summary
`data/run-reports/phase-6a-rag-holdout-r03-summary.json`, whose SHA-256 is
`4f193f50d5f162e716bdbc7c244e2faa77086d3ea5d48bee0c5f04cfabe20d88`. The record binds the frozen
SUT commit, controlled freeze, execution success, final metrics, failed cases, privacy boundary,
and future-evaluation status without copying queries, responses, prompts, quotes, evidence text,
vectors, secrets, request IDs, or provider exception details.

The deterministic internal/public citation-ID mapping defect is independently proven to exist in
the SUT. R03 did not preserve the exact per-response internal-to-public citation trace, so its causal
contribution to `HOLD-P04` or `HOLD-P05` is plausible but not conclusively established by the
persisted R03 evidence.

R03 must never be treated or rerun as a fresh holdout. After any SUT remediation, unbiased quality
validation requires a new R04 fresh holdout. R03 may thereafter be referenced only as finalized,
exposed regression evidence; its operational reports, decision, metrics, and history remain
immutable.
