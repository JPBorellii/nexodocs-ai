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
