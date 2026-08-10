# Answerability calibration attempt lifecycle contract v1

## Status and governing scope

This document is the versioned attempt-lifecycle addendum for the first
real-provider answerability development-calibration baseline. The frozen baseline
contract at commit `7b76f6016e500fdd5451ce3d8ca9e711428a2a4f` and this addendum jointly
govern execution.

The addendum amends only the authorized persisted artifacts and the lifecycle
semantics of a baseline attempt. It does not change the model, reasoning effort,
token limit, timeout, transport retries, prompt, oracle, corpus, semantic quality
thresholds, technical-validity thresholds, support metric role, R03/R04 policy,
or the rule that the first technically valid attempt is the canonical quality
baseline.

The machine-readable source of truth is
`evals/rag/answerability-calibration-attempt-lifecycle-contract-v1.json`. Each
journal event is governed by
`evals/rag/answerability-calibration-attempt-lifecycle-event-v1.schema.json`.

## Why the addendum is required

The complete calibration result v1 requires exactly 32 cases and is created only
after evaluation finishes. It cannot truthfully represent a process that stops
before all cases complete, and it does not carry a baseline attempt number. Those
properties are appropriate for a complete result, so its schema remains unchanged.

This addendum authorizes exactly one additional authoritative persisted artifact
type: a privacy-safe, append-only attempt lifecycle journal. The journal preserves
reservation, preflight, case progress, result publication, termination, and
recovery state without storing model inputs or raw provider outputs. It supplements
the complete result; it never replaces it.

## Frozen artifact paths

Attempt numbers are positive integers from 1 through 9999, rendered as exactly
four zero-padded decimal digits. The lifecycle path is:

`evals/rag/answerability-calibration-baseline-attempt-NNNN.lifecycle.jsonl`

The complete result path is:

`evals/rag/answerability-calibration-baseline-attempt-NNNN.result.json`

For example, attempt 1 uses
`answerability-calibration-baseline-attempt-0001.lifecycle.jsonl` and
`answerability-calibration-baseline-attempt-0001.result.json`. Attempt 2 uses the
corresponding `0002` paths. No attempt artifact may overwrite an existing path.

## Exact attempt start boundary

An attempt has not started while the launcher validates the clean committed
worktree, frozen contracts and bindings, prior attempts, target paths, and the
presence of a non-empty `OPENAI_API_KEY`. A missing or empty key therefore does
not consume an attempt number.

An attempt starts at the instant its first `attempt_reserved` event has been
created through exclusive journal creation, fully written, flushed, and durably
synchronized. Durable reservation must happen before concrete provider
construction, the preflight provider request, and every oracle case request.
Once reservation is durable, the attempt number is consumed forever, even if the
next operation fails.

This boundary makes provider use impossible without a prior durable audit record.
It also prevents configuration or missing-key failures from creating misleading
attempts.

## Append-only journal semantics

The journal is JSON Lines encoded as UTF-8 without BOM and with LF line endings.
Every line is one object accepted by the closed event schema. Sequence starts at
1 and increases by exactly 1. Existing bytes may never be rewritten, deleted,
truncated, or replaced.

Each successfully appended event must be flushed and durably synchronized. An
event that announces a started operation must complete that synchronization
before the operation begins. In particular, `preflight_started`, `case_started`,
and `result_publication_started` precede the operations they announce. A valid
journal prefix followed by process termination is itself auditable evidence. A
malformed or truncated trailing line must be reported; a validator may not
silently discard the valid earlier events.

The only event types are:

- `attempt_reserved`
- `preflight_started`
- `preflight_succeeded`
- `preflight_failed`
- `case_started`
- `case_completed`
- `result_publication_started`
- `result_published`
- `attempt_terminated`

Every event binds the attempt number, sequence, frozen baseline contract commit
and JSON hash, and the clean committed SUT HEAD. The SUT commit is captured at
runtime rather than hard-coded and remains identical throughout one journal.

`case_started` stores only `case_id`. `case_completed` stores only `case_id` and
the sanitized operational status beyond the common fields. The journal contains
no token usage; safe usage metadata remains owned by the complete result v1.

## Incomplete, failed, and terminal attempts

An `attempt_terminated` event has one of three statuses:

- `technically_valid`: a complete 32-case result exists and passes every frozen
  technical-validity gate. Quality is authoritative. If no earlier technically
  valid attempt exists, this is the canonical quality baseline.
- `technically_invalid`: execution reached a conclusive technical failure.
  Quality is not authoritative.
- `interrupted`: execution ended without a complete authoritative technical
  result. Quality is not authoritative.

A journal without `attempt_terminated` is nonterminal. It is not automatically
eligible for retry. A crash may leave `case_started` without `case_completed`, or
`result_publication_started` without `result_published`; these valid prefixes
identify the incomplete operation without exposing its inputs.

## Concurrency and explicit recovery

The future launcher must hold an OS-managed exclusive advisory lock on the
lifecycle journal throughout the live attempt. The journal itself is the
persistent reservation artifact. This contract does not authorize a second
persistent lock file.

When a later process inspects a nonterminal journal, failure to acquire the
exclusive journal lock means the prior attempt is live and retry is forbidden.
Successful lock acquisition only establishes that the attempt is an abandoned
candidate. It does not abandon the attempt automatically.

Recovery requires an explicit operator action. Recovery performs zero provider
calls and appends exactly one `attempt_terminated` event with status
`interrupted`, reason `recovered_abandoned_attempt`, and
`quality_authoritative=false`. It never rewrites prior events. The recovered
journal must be committed before another attempt begins, because every real
attempt starts from a clean committed worktree.

## Sequential and canonical attempts

If no journal exists, only attempt 1 is allowed. Otherwise the next attempt must
be exactly the highest validated attempt plus one, and all attempt numbers must be
contiguous. Lifecycle and result targets must both be absent before reservation.

There are zero automatic reruns. A prior technically invalid attempt or an
explicitly recovered interrupted attempt permits only the explicit next number.
A live nonterminal attempt blocks the next attempt, and an abandoned nonterminal
attempt must first be explicitly recovered.

The first technically valid attempt blocks every later attempt even when it fails
the frozen quality gates. An operator may never choose a later attempt because it
has better quality.

## Result binding and publication

The complete result remains governed by
`evals/rag/answerability-calibration-result-v1.schema.json`; that schema does not
need an `attempt_number`. The journal binds a result to its attempt using the
attempt number, result-relative path, result SHA-256, SUT commit, and frozen
baseline contract binding.

`result_published` may be appended only after the complete privacy-safe result has
been locally validated and successfully published. A technically valid terminal
event, or a technically invalid terminal event for a completed 32-case run, must
agree with the bound result.

Publication begins from fully serialized and locally validated privacy-safe
bytes, never overwrites an existing target, and fails closed. A truncated,
partially written, or schema-invalid result is never authoritative. If publication
fails, the journal remains sufficient to preserve the consumed attempt. The exact
atomic-publication mechanism is deferred to the implementation code review, and
no raw or sensitive temporary content is authorized.

## Git provenance invariant

Each real attempt begins from clean committed HEAD `S`, and every event records
`sut_commit = S`. After execution or recovery, only authorized attempt artifacts
may remain uncommitted, and they must be committed before another attempt starts.
For the original execution artifacts, preservation commit `R` must have
`R^ == S`.

Recovery does not rewrite provenance. All original events retain their immutable
`sut_commit`; recovery appends only the terminal recovery event before the
recovered artifact is committed. No future implementation commit SHA is
hard-coded in source.

## Privacy boundary

The lifecycle journal must never persist API keys, GitHub tokens, raw queries,
raw evidence, supporting quotes, raw provider responses, raw refusals, rendered
prompts, system prompts, provider request IDs, SDK exception text, environment
dumps, or token usage.

It may persist only the closed schema fields: attempt and sequence numbers, event
type, case ID where applicable, sanitized operational status, fixed sanitized
failure or terminal reason, frozen contract bindings, SUT commit, result-relative
path and SHA-256 where applicable, and the quality-authoritative boolean.

## Implementation readiness

This lifecycle contract is ready for provider, launcher, journal, recovery, and
offline-test implementation. Real execution remains blocked. It may become ready
only after the explicit reasoning-effort provider change and the dedicated
launcher/lifecycle implementation pass code review and all directed offline
tests. This contract does not authorize OpenAI execution by itself.
