# Answerability Calibration Attempt Lifecycle — Crash & Recovery Addendum v1

- **Contract ID:** `answerability-calibration-attempt-lifecycle-crash-recovery-addendum-v1`
- **Version:** `1.0.0`
- **Status:** Normative addendum
- **JSON SHA-256:** `62db0ffbdbf53a270a5538b1f11e82e2524f76636f1045329583625be52d1379`

## 1. Purpose

This addendum narrows and hardens only the crash, torn-reservation, result-visibility,
and abandoned-attempt recovery semantics of the frozen answerability baseline lifecycle.
It does **not** redefine the oracle, the 32 cases, the result schema, the event schema,
the execution profile, the provider, the model, the prompt, the quality gates, R03, or R04.

Its central integrity goal is to prevent a user or operator from learning result material
and then obtaining a fresh provider attempt merely because a terminal lifecycle event was
lost or a publication operation was interrupted.

## 2. Base contract and immutable bindings

This addendum is governed by the already-frozen lifecycle contract:

- JSON: `evals/rag/answerability-calibration-attempt-lifecycle-contract-v1.json`
- JSON SHA-256: `2892bab503e9cfe9e314fff4fd1e2d63eb41e80aa8e7ba83c7ca4b98f688114d`
- Markdown: `docs/evaluation/answerability-calibration-attempt-lifecycle-contract-v1.md`
- Markdown SHA-256: `be67b5b558e11e311beb278b196d69343293bbdd90f8dabb7f88c3ba7d5dd111`

Additional governing bindings remain:

- Baseline contract JSON SHA-256:
  `42bdf970b3231c7eb1f70c7bbfe19d8cd51e660de85ae97388105e4b94f611ca`
- Baseline contract commit:
  `7b76f6016e500fdd5451ce3d8ca9e711428a2a4f`
- Lifecycle event schema SHA-256:
  `80df5b4a285a1aa0bfedc77ed43560aee70492d146123121905897a6ef100375`
- Result schema SHA-256:
  `94b1aa8f0cf6b87335982f857103d88d1c71f07681441cef4482451f2cadbbca`
- Development oracle SHA-256:
  `cb39f40f1d13cb09ecd042d81d99299eb2bcb0657602c47c5dcd2cc7d0d4d216`

The original event schema and result schema remain unchanged.

## 3. Narrow precedence

This addendum overrides the base lifecycle contract only where the original rules are
insufficient to decide safely after an abrupt failure or after result bytes become
potentially inspectable.

It overrides only:

1. interpretation of a persisted `attempt_reserved` record after abrupt loss;
2. cleanup semantics for ordinary pre-start failures in the creating process;
3. live failure classification after result bytes become potentially inspectable;
4. explicit recovery when attempt-scoped result bytes survive;
5. recovery event count when `result_published` must be reconstructed;
6. recovery lineage when recovery deterministically appends `result_published` and a
   bound terminal.

Everything else remains governed by the original frozen lifecycle contract.

## 4. Attempt-start boundary remains unchanged

The original attempt-start boundary is preserved.

An attempt starts only after its first `attempt_reserved` event has been:

1. created through exclusive lifecycle-path creation;
2. fully written;
3. flushed;
4. durably synchronized.

That boundary must still precede concrete provider construction, preflight, and every
oracle request.

### 4.1 Ordinary failure in the same creating process

If an ordinary failure occurs before the process successfully crosses the start
boundary, the provisional lifecycle artifact may be removed **only** when the process
can still establish that it owns the descriptor created by exclusive creation and that
the current path has not been observed as a replacement.

If ownership or path identity cannot be established, cleanup fails closed.

A detected replacement is never deleted.

A successfully cleaned pre-start failure consumes no attempt number.

### 4.2 Later process sees a valid reservation record

After abrupt process loss, a later process cannot prove whether the historical fsync
returned before the crash merely by observing a complete `attempt_reserved` line.

Therefore this addendum adopts a conservative safety rule:

> If the later process can parse and fully validate a journal beginning with the frozen
> `attempt_reserved` event, the attempt number is treated as consumed.

This may conservatively consume an attempt that historically failed immediately before
the fsync returned. That availability cost is intentional. Reusing a possibly consumed
attempt number is considered the greater integrity risk.

A valid abandoned journal may then follow the applicable explicit-recovery rules below.

### 4.3 Later process sees an ambiguous torn reservation artifact

If the lifecycle path exists but is empty, malformed, truncated, or otherwise cannot
be validated as a journal beginning with `attempt_reserved`, the start boundary cannot
be reconstructed safely.

In that state:

- provider calls are forbidden;
- automatic deletion is forbidden;
- rewrite or truncation is forbidden;
- same-number reuse is forbidden;
- a higher real attempt is forbidden;
- automatic recovery is forbidden.

The baseline series is **blocked** until a separately governed forensic/manual
resolution policy exists. This addendum intentionally defines no destructive repair
procedure.

## 5. Result-visibility boundary

This addendum introduces a safety boundary distinct from the original
`result_published` event.

For live-failure and recovery decisions, result bytes are considered potentially
inspectable once either of these attempt-scoped paths exists:

- canonical result:
  `evals/rag/answerability-calibration-baseline-attempt-NNNN.result.json`
- deterministic temporary result:
  `evals/rag/.answerability-calibration-baseline-attempt-NNNN.result.tmp`

The existence of potentially inspectable result bytes changes which terminal states are
safe.

## 6. Live failures before result visibility

If result publication fails and **neither** the attempt-scoped temporary artifact nor
the canonical result artifact survives, the existing unbound terminal remains legal:

- `terminal_status = technically_invalid`
- `terminal_reason = result_publication_failed`
- `quality_authoritative = false`
- no result binding

This case does not leave surviving result bytes that an operator could inspect before
deciding whether to request another provider attempt.

## 7. Live failures at or after result visibility

Once result bytes survive at either attempt-scoped result path, the launcher must not
convert the attempt into an unbound retry-eligible terminal.

At or after the result-visibility boundary, the following unbound terminals are
forbidden:

- `interrupted / operator_interrupted`
- `interrupted / system_exit`
- `technically_invalid / unexpected_execution_failure`
- `technically_invalid / result_publication_failed`

If the live process can deterministically finish the already-produced result safely, it
may do so. Otherwise it leaves the journal nonterminal and requires explicit
zero-provider recovery.

No automatic rerun is ever allowed.

## 8. Explicit recovery — common rules

Every explicit recovery:

- requires explicit operator action;
- acquires the original journal lock exclusively;
- performs zero provider calls;
- performs zero network calls;
- does not read `OPENAI_API_KEY`;
- preserves the original `sut_commit`;
- requires current HEAD to equal that original SUT commit;
- permits only authorized attempt artifacts in the worktree;
- never rewrites existing lifecycle bytes;
- never overwrites an existing canonical result;
- never re-runs quality evaluation to choose a preferable outcome;
- uses only the frozen technical-validity gate when a complete result survives.

## 9. Recovery state matrix

### 9.1 Pre-publication, no surviving result artifacts

Conditions:

- journal is valid and nonterminal;
- no `result_published` event;
- no attempt-scoped temporary result;
- no canonical result.

Recovery appends exactly one event:

- `attempt_terminated`
- `terminal_status = interrupted`
- `terminal_reason = recovered_abandoned_attempt`
- `quality_authoritative = false`
- no result binding

After preservation commit, the existing sequential policy may permit the explicit next
attempt.

### 9.2 Valid temporary result only

Conditions:

- `result_publication_started` already exists;
- no `result_published` event;
- attempt-scoped temporary result exists;
- canonical result does not exist.

Recovery must:

1. validate the temporary bytes as the complete frozen result;
2. establish the canonical result using exactly those validated bytes, with no provider
   call and no overwrite;
3. synchronize and re-read the canonical result;
4. compute SHA-256 from the re-read canonical bytes;
5. append `result_published`;
6. compute only frozen technical validity;
7. append the matching bound terminal.

If the temporary result cannot be validated or safely promoted, recovery fails closed
and the series is blocked.

### 9.3 Valid canonical result, missing `result_published`

Conditions:

- `result_publication_started` already exists;
- canonical result exists;
- `result_published` is absent.

Recovery must:

1. validate the canonical bytes as the complete frozen result;
2. synchronize and re-read them;
3. compute SHA-256;
4. append `result_published`;
5. compute only frozen technical validity;
6. append the matching bound terminal.

The operator has no choice over the terminal result.

### 9.4 Valid canonical result with `result_published`

Recovery verifies:

- event path equals the exact attempt result path;
- event SHA-256 equals the canonical bytes;
- canonical bytes validate against the frozen result contract;
- result case IDs and operational statuses agree with the lifecycle journal.

Recovery then appends **only** the matching bound terminal.

### 9.5 Temporary and canonical results both survive

Both artifacts must validate and be byte-identical.

If they differ, recovery fails closed and the series is blocked.

If they are identical, the canonical bytes govern. `result_published` is appended only
when missing, followed by the matching bound terminal. Temporary cleanup may occur only
under the existing best-effort identity safeguards. Failure to clean safely prevents
terminal completion.

### 9.6 Inconsistent or unverifiable result artifact

Examples include:

- invalid temporary result;
- invalid canonical result;
- differing temporary and canonical bytes;
- `result_published` digest mismatch;
- lifecycle/result case mismatch;
- unreadable or unsafely synchronizable result artifact.

In this state:

- provider calls remain zero;
- no recovery terminal is appended;
- no later real attempt is allowed;
- the series is blocked.

This is intentionally stricter than converting the attempt into a retry-eligible
technical failure after potentially inspectable result material exists.

## 10. Deterministic recovered terminal

A surviving complete result can produce only one of two result-bound terminals.

### Technically valid

- `terminal_status = technically_valid`
- `terminal_reason = technical_gates_passed`
- `quality_authoritative = true`
- result path and digest are required

If no earlier technically valid attempt exists, this result is canonical.

A quality failure does **not** permit another attempt.

### Technically invalid complete result

- `terminal_status = technically_invalid`
- `terminal_reason = technical_gates_failed`
- `quality_authoritative = false`
- result path and digest are required

Only the existing frozen technical-validity gate determines this state. Quality
thresholds do not participate in the recovery decision.

A later explicit attempt is permitted only under the existing sequential policy and
only after the recovered artifacts are preserved in a commit.

## 11. Result-shopping prevention invariant

Once complete result bytes survive in an attempt-scoped result artifact, an operator
may not:

- discard a valid result to request a fresh provider attempt;
- reclassify a valid result as `interrupted`;
- reclassify a valid result as unbound `unexpected_execution_failure`;
- reclassify a valid result as unbound `result_publication_failed`;
- choose the terminal state based on quality metrics.

If deterministic recovery establishes a technically valid result whose quality gates
fail, that result still becomes the first canonical technically valid baseline and
blocks every later attempt.

## 12. Recovery event-count override

The base rule of one appended event remains true for pre-publication interrupted
recovery.

For deterministic result recovery, this addendum authorizes at most two appends:

1. optional `result_published`, only when a valid canonical result exists or is
   deterministically established and the event is missing;
2. exactly one bound `attempt_terminated`.

No other event synthesis is authorized.

No new event type is introduced.

## 13. Recovery provenance

All recovery events retain the original immutable `sut_commit` and frozen bindings.

For result recovery, the only authorized recovery append sequence is:

1. optional `result_published`;
2. required bound `attempt_terminated`.

Recovered artifacts must be committed before any later attempt.

This addendum does not authorize rewriting, deleting, truncating, or replacing earlier
journal events.

## 14. Privacy

The existing privacy boundary remains unchanged.

Recovery must not persist API keys, GitHub tokens, raw queries, raw evidence, raw
provider responses, raw refusals, rendered prompts, system prompts, provider request
IDs, SDK exception text, or environment dumps.

An attempt-scoped result temporary artifact may contain only the same already-governed
privacy-safe complete result shape intended for the canonical result.

## 15. Trusted local-host boundary and residual TOCTOU

The implementation assumes the repository working directory is not being deliberately
raced by a malicious local process.

Path/inode identity checks remain best-effort race detection. This addendum does not
claim a portable atomic compare-and-unlink primitive across Windows and POSIX.

The residual local TOCTOU risk is accepted and documented under that trusted-host
boundary; detected path replacement must still fail closed.

## 16. Implementation requirements

Before any real attempt, implementation must:

- bind this addendum by exact SHA-256;
- validate the base lifecycle contract and governing hashes;
- preserve the existing event schema and result schema;
- keep one OS-managed journal lock and create no second persistent lock file;
- never call the provider from recovery;
- never automatically rerun a real attempt;
- never overwrite lifecycle or canonical result artifacts;
- never append an unbound terminal after the result-visibility boundary;
- never silently delete an ambiguous torn reservation artifact;
- block later attempts after any `series_blocked` classification;
- use only frozen technical validity to determine a recovered bound terminal;
- ignore quality metrics when deciding recovered technical validity;
- validate Git HEAD and authorized worktree scope before and after recovery appends.

## 17. Required acceptance tests

The implementation is not approved until tests prove, at minimum:

1. ordinary zero-byte, partial-write, first-fsync, and directory-sync pre-start failures;
2. owned provisional cleanup without attempt consumption;
3. detected replacement is never deleted;
4. valid post-crash `attempt_reserved` is conservatively consumed;
5. empty or malformed torn reservation blocks with zero provider calls;
6. pre-visibility publication failure may use unbound `result_publication_failed`;
7. any surviving result artifact forbids unbound interruption/failure terminals;
8. valid temp-only recovery finalizes deterministically with zero provider calls;
9. valid canonical-no-event recovery reconstructs `result_published` and finalizes;
10. valid canonical-with-event recovery appends only the matching bound terminal;
11. temp/canonical mismatch blocks the series;
12. invalid or unverifiable surviving result blocks the series;
13. recovered technically valid quality-pass blocks later attempts;
14. recovered technically valid quality-fail also blocks later attempts;
15. recovered technically invalid complete result follows only the existing sequential
    next-attempt policy after preservation commit;
16. `KeyboardInterrupt`, `SystemExit`, and ordinary exceptions after result visibility
    cannot create an unbound retry-eligible terminal;
17. recovery performs zero provider or network calls and reads no OpenAI key;
18. journal bytes remain append-only;
19. canonical result is never overwritten;
20. every recovery append preserves the original SUT and frozen bindings;
21. privacy sentinels never enter lifecycle or unauthorized operator output.

## 18. Readiness

This addendum is semantically complete and implementation-ready.

It does **not** authorize real execution.

Real execution remains blocked until:

1. the exact addendum bytes are frozen and committed;
2. implementation binds and enforces the addendum;
3. directed answerability/lifecycle tests pass;
4. Ruff passes;
5. Pyright strict passes;
6. `git diff --check` passes;
7. full pytest causality is reviewed;
8. the final independent pre-execution audit passes;
9. explicit human authorization is given for the one real non-oracle preflight.
