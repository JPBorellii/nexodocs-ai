# Full RAG holdout R01 incident and R02 preflight

## R01 technical result

`full-rag-holdout-r01` is permanently recorded as `TECHNICALLY_FAILED` and
`NOT_EVALUATED`. The run stopped at `HOLD-P01` in the answer provider with HTTP 400, produced no
stdout JSON, and created neither a usage report nor an evaluation summary. Because those reports do
not establish call counts, both observed call fields are `null` rather than zero.

The sanitized incident artifact, closed JSON Schema, and offline validator are:

- `evals/rag/full-rag-holdout-r01-technical-incident.json`
- `evals/rag/full-rag-holdout-r01-technical-incident.schema.json`
- `scripts/validate_full_rag_holdout_r01_technical_incident.py`

The probable, not confirmed, root cause is
`OPENAI_STRUCTURED_OUTPUT_SCHEMA_INCOMPATIBILITY`. The artifact does not retain queries, answers,
tracebacks, provider messages, request IDs, timestamps, absolute paths, or secrets.

## R02 preflight

`full-rag-holdout-r02` is a new, unexecuted attempt. Its fixture preserves all 12 R01 cases and
expectations and links R01 by fixture SHA-256. Its system freeze records the corrected provider
boundary while retaining the frozen prompt, threshold, retrieval, grounding, citation validator, and
safety preflight hashes.

Run the four offline checks before any authorized execution:

```powershell
uv run --locked python scripts/validate_full_rag_holdout_r01_technical_incident.py
uv run --locked python scripts/validate_openai_structured_output_schema.py
uv run --locked python scripts/validate_full_rag_holdout_fixture.py --attempt r02
uv run --locked python scripts/validate_full_rag_system_freeze.py --attempt r02
```

No R02 operational report is versioned or generated here. Future reserved report names are
`phase-6a-rag-holdout-r02-p01.json` through `n06.json` and
`phase-6a-rag-holdout-r02-summary.json` under `data/run-reports/`.

## R02 adjudication and sanitized diagnostic preparation

R02 was later executed and adjudicated as `FULL_RAG_HOLDOUT_FAILED`. The adjudicated rates are:

- supported grounded answers: `0.33333333`;
- supported expected-document citations: `0.50000000`;
- supported citation validity: `1.00000000`;
- supported required-fact coverage: `0.33333333`;
- unsupported safe fallback responses: `0.83333333`;
- unsupported hallucinations: `0.00000000`;
- total case safety: `0.58333333`;
- schema validity: `1.00000000`;
- clinical safety: `true`;
- prompt or secret leakage count: `0`.

`HOLD-P02`, `HOLD-P03`, `HOLD-P04`, and `HOLD-N04` had technical grounding failures. P05 is
formally recorded in `evaluation-oracle-corrections-v1.json` as
`EVALUATION_ORACLE_DEFECT`, scoped to the evidence retrieved in R02. The observed evidence did not
confirm active Nexo Integral coverage for the North unit; it associated Nexo Integral with the
Center unit and an observed `restricted` status. This is a sanitized finding, not a reproduced
excerpt or a rewrite of the canonical source.

The new closed `safe_error_code` taxonomy only improves observability. It does not change grounding
strictness or any answer semantics. The `grounding-diagnostic-d03` schema and offline validator are
ready, but D03 has not been executed and no R03 fixture, freeze, report, or summary exists.
