# Answerability calibration real-provider baseline contract v1

## Status and purpose

This document freezes the acceptance and execution contract for the first real-provider answerability development baseline. It was created before any real baseline result was observed. The bound fixture is development calibration data: it is neither R03 nor R04 and it is not an unseen holdout.

Freezing the criteria first prevents result-aware threshold selection. No threshold in this contract may be changed after the first real baseline is observed. A later change requires a separately versioned contract and must not be used to relabel the frozen baseline result.

The machine-readable source of truth is `evals/rag/answerability-calibration-baseline-contract-v1.json`.

## Frozen bindings

- Base commit: `3b375cc303739bdfac80d9cd8072967c7127d836`.
- Fixture: `nexodocs-answerability-calibration-v1`, schema `1.0.0`, SHA-256 `cb39f40f1d13cb09ecd042d81d99299eb2bcb0657602c47c5dcd2cc7d0d4d216`.
- Fixture composition: exactly 32 cases, with 16 `answerable` and 16 `insufficient` cases.
- Canonical corpus SHA-256: `a49ad2a4c3b8dc767277566be60e93cd2a8d3113ffe1c598eec9ccfb759b02c4`.
- Classifier prompt: `answerability-v1`, SHA-256 `0b9c69791182b67603342070ea7232ade4d80c2e7818ce4f2cb37ca0c3e27529`.

No raw oracle query or evidence text is reproduced here.

## Semantic boundary

`answerable` means that the supplied evidence supports every material fact and constraint requested by the query. `insufficient` means that at least one requested material fact or constraint is not supported, even when the evidence is topically related or makes an answer seem plausible.

False answers receive the stricter gate because they cross the safe abstention boundary: they assert answerability where the evidence is incomplete. A false abstention reduces usefulness, while a false answer can cause the downstream system to generate an unsupported answer. The asymmetry is intentional and frozen.

## Frozen execution profile

The provider is OpenAI through the Responses API, using model `gpt-5.6-luna` with explicit reasoning effort `medium`, `max_output_tokens` 300, timeout 30.0 seconds, zero transport retries, and `store` set to false. Output must use strict JSON Schema Structured Outputs.

Each of the 32 cases must be a separate provider request. There is no `previous_response_id`, conversation carry-over, or cross-case state.

## Technical validity gate

Technical validity is decided before model quality is interpreted. A run is technically valid only when all of the following hold:

- Exactly 32 cases were evaluated.
- All 32 cases have `operational_status == success`.
- `provider_refusal_count == 0`.
- `provider_failure_count == 0`.
- `invalid_output_count == 0`.
- `evaluator_failure_count == 0`.

If any condition fails, the disposition is `BASELINE_EXECUTION_INVALID`. Partial quality metrics from that run must not be interpreted as an accepted baseline.

## Baseline attempt and rerun policy

Attempt 1 and every later started attempt must be preserved as a separately numbered, non-overwritable artifact. There are no automatic full-baseline reruns. A technically invalid attempt has no authoritative quality result, but it may be retried only as an explicit new attempt so that provider refusals, provider failures, invalid outputs, evaluator failures, and incomplete case executions remain auditable rather than being silently discarded.

Every new attempt must use the same frozen contract, execution profile, oracle, corpus, classifier, and acceptance criteria. The first technically valid attempt is the canonical quality baseline. Once it exists, no later run may replace it or be selected because it has better quality; operators must never choose among multiple technically valid attempts.

## Quality acceptance gate

Quality is evaluated only for a technically valid run, and PASS requires every condition below.

Overall:

- `passed_cases >= 28/32`.
- `overall_decision_accuracy >= 0.875` (28 divided by 32).

Answerable cases:

- `answerable_total == 16`.
- `answerable_correct >= 13/16`.
- `answerable_recall >= 0.8125` (13 divided by 16).
- `false_abstention_count <= 3`.
- `false_abstention_rate <= 0.1875` (3 divided by 16).

Insufficient cases:

- `insufficient_total == 16`.
- `insufficient_correct >= 15/16`.
- `insufficient_recall >= 0.9375` (15 divided by 16).
- `false_answer_count <= 1`.
- `false_answer_rate <= 0.0625` (1 divided by 16).

Support ID agreement is diagnostic only. It may help analyze which evidence the classifier selected, but it must not determine semantic baseline PASS or FAIL.

## Development calibration and future evaluation

This fixture remains development calibration data and may be used for development calibration only. R03 is already exposed: its content and results are prohibited for tuning this classifier, and R03 must not be rerun for calibration. R04 remains the required fresh unseen holdout after SUT remediation or integration. Performance on this fixture must not be presented as R04 or as unseen-holdout performance.

## Real-provider provenance

A `provider_name == "openai"` string does not prove real-provider provenance because an injected fake can declare the same value. The future `real_provider_baseline` launcher must directly construct the project's concrete `OpenAIAnswerabilityProvider`, reject arbitrary provider injection, and bind the exact frozen model and execution profile.

The launcher must obtain the API key only from the user-controlled process environment and must never print or persist it. Before starting any of the 32 cases, it must perform a non-oracle model-access preflight and fail closed if provider provenance or configuration validation fails. Its only persisted output may be the privacy-safe calibration result contract.

## Execution readiness

`execution_ready` is frozen as `false`. At this commit, `OpenAIAnswerabilityProvider` does not expose an explicit reasoning-effort parameter and does not send the frozen `medium` value in its Responses API request. The baseline must not run until a minimal, separately reviewed provider/launcher change can explicitly enforce `reasoning_effort = medium` and the launcher satisfies every provenance, configuration, preflight, isolation, secret-handling, and privacy-safe-output requirement above.
