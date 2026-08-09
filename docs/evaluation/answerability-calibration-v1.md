# Answerability Calibration Oracle v1

## Status and purpose

This document freezes the human adjudication record for the NexoDocs AI pre-implementation answerability calibration fixture. The machine-readable oracle is `evals/rag/answerability-calibration-v1.json`.

The fixture is development/calibration data. It is not R04, is not a retrieval benchmark, and must not be promoted or reused as an unseen holdout.

## Answerability boundary

For this oracle, **answerable** means that the evidence set supplied with a case supports every material fact and constraint requested by its query. The evidence is sufficient to attempt an answer; the oracle does not assert that a later generated answer will be well written, complete, or correctly cited.

**Insufficient** means that the evidence can be related, plausible, or lexically similar while still lacking at least one requested material fact or constraint. A document title, topic match, familiar entity, or high superficial relevance does not make a case answerable.

Adjudication is based on the exact evidence set declared by `evidence_chunk_ids`, not on other chunks that happen to exist elsewhere in the corpus.

## Out of scope

This oracle does not evaluate:

- retrieval order, ranking, scores, or thresholds;
- public citation-ID provenance;
- final-answer prose or completeness;
- grounding after generation;
- clinical preflight;
- prompt-injection preflight;
- provider behavior, embeddings, or vector-store behavior.

No retrieval threshold, lexical cutoff, score rule, classifier prompt, provider implementation, or model-generated reasoning expectation is part of the fixture.

## Corpus provenance

The fixture binds to `knowledge_base/processed/chunks.jsonl` with SHA-256 `a49ad2a4c3b8dc767277566be60e93cd2a8d3113ffe1c598eec9ccfb759b02c4`. This value was recomputed from the artifact and verified against `knowledge_base/processed/manifest.json`, which declares the same canonical hash and a total of 44 chunks.

Each case references canonical chunk IDs rather than duplicating chunk text. Its `evidence_provenance` binds every referenced ID to that chunk's canonical `text_sha256`, making the exact evidence set reconstructible and tamper-detectable.

Fixture schema version is `1.0.0`; canonical processed-chunk schema version is `1.0`.

## Independence policy

The oracle was designed before any answerability classifier prompt or provider implementation. Construction used only the canonical fictional knowledge base, its processed manifest/schema, and current production RAG contracts.

R02 and R03 material was prohibited because exposed holdout cases, outcomes, or failure analyses would leak evaluation targets into the calibration oracle. No exposed holdout case content or diagnostic material was used. The machine-readable declaration sets `exposed_holdout_used` to `false`.

## Construction method

All 44 canonical chunks were reviewed directly. For every answerable case, the adjudicator verified that the declared supporting chunk or union of supporting chunks covered all material facts and constraints. For every insufficient case, the adjudicator verified that at least one material requested fact or constraint was absent from the supplied evidence.

Questions are independent, fictitious formulations created from allowed corpus content. Negatives deliberately retain nearby domain context or shared vocabulary so that the oracle measures semantic sufficiency rather than keyword detection. Positives include nontrivial paraphrase and cases in which a strong supporting chunk appears alongside weaker evidence.

Cases marked `requires_collective_multi_chunk_support: true` satisfy a stricter rule: no single declared supporting chunk answers the complete query, while the union does.

Evidence remains untrusted data even when it contains imperative or directive language. ACAL-A06 and ACAL-I16 exercise this property using existing canonical policy text; the text must inform the sufficiency decision without being followed as an instruction to the classifier.

## Category design

The eight categories contain four cases each:

- `answerable_single_chunk_clear` establishes unambiguous positive anchors supported by one chunk.
- `answerable_collective_multi_chunk` tests complementary evidence whose union is required.
- `answerable_paraphrase_indirect` tests synonymy, indirect wording, and lexical mismatch.
- `answerable_material_constraint` tests supported numeric, date, entity, status, and policy constraints.
- `insufficient_near_domain_absent_fact` tests relevant domains whose requested fact is absent.
- `insufficient_shared_vocabulary_missing_fact` tests topic and keyword overlap without factual support.
- `insufficient_missing_material_constraint` tests missing numeric, date, entity-combination, status, or guarantee constraints.
- `insufficient_plausible_business_question` tests credible corporate questions that the supplied evidence cannot answer.

Categories are primary construction strata. `coverage_tags` record useful overlapping properties without changing the fixed category counts.

## False-abstention risk

An overly conservative classifier can abstain when wording differs from the evidence, when several chunks must be combined, or when weak distractors accompany one strong chunk. Such false abstentions would suppress legitimate answers even though the complete requested facts are supported. ACAL-A05 through ACAL-A12 specifically guard these failure modes, including multi-chunk composition, paraphrase, indirect negation, and strong support mixed with weak evidence.

The opposite error is also controlled: related policy language, entity overlap, or plausible corporate context must not license invented details. The sixteen insufficient cases require abstention precisely because at least one material constraint is missing.

## Interpretation of expected support

For answerable cases, `expected_support_chunk_ids` lists only chunks that materially support the requested answer. It may be a strict subset of `evidence_chunk_ids`; retrieved but irrelevant or merely weak chunks are not required support.

For insufficient cases, `expected_support_chunk_ids` is always empty. This does not claim the evidence is unrelated. It means the complete requested answer lacks sufficient support, so the oracle does not designate partial evidence as support for an answer attempt.

`rationale_for_human_review`, `expected_decision`, `expected_support_chunk_ids`, and `coverage_tags` are adjudication labels. They must never be provided as classifier input. The intended future classifier input consists only of the query and the evidence reconstructed from `evidence_chunk_ids`.

## Human adjudication ledger

| Case | Decision | Category | Human finding |
|---|---|---|---|
| ACAL-A01 | answerable | single chunk | Minimum advance of 24 hours is explicit. |
| ACAL-A02 | answerable | single chunk | Thirty calendar days and the People channel occur together. |
| ACAL-A03 | answerable | single chunk | The privacy reporting area and channel are explicit. |
| ACAL-A04 | answerable | single chunk | The exact unit/plan row provides status and confirmation. |
| ACAL-A05 | answerable | collective | One chunk supplies advance/channel; another supplies response time. |
| ACAL-A06 | answerable | collective | One chunk supplies both required pre-generation actions; another supplies the reporting channel. |
| ACAL-A07 | answerable | collective | Separate rows are required to compare two units. |
| ACAL-A08 | answerable | collective | Separate directory rows are required for two areas. |
| ACAL-A09 | answerable | paraphrase | No-show and monetary charge map to absence and financial penalty. |
| ACAL-A10 | answerable | paraphrase | The evidence directly negates automatic approval in different wording. |
| ACAL-A11 | answerable | paraphrase | The strong chunk supports responsible area and exclusion of values despite weak extra evidence. |
| ACAL-A12 | answerable | paraphrase | Personal email/public link map to personal channels/public URLs. |
| ACAL-A13 | answerable | material constraint | Exact policy date and responsible area are present. |
| ACAL-A14 | answerable | material constraint | Exact date, entity, status, and confirmation are present. |
| ACAL-A15 | answerable | material constraint | Exact contact, response time, and directory status are present. |
| ACAL-A16 | answerable | material constraint | The count and no-cost condition are explicit. |
| ACAL-I01 | insufficient | near domain | Cancellation evidence lacks a phone or WhatsApp number. |
| ACAL-I02 | insufficient | near domain | Vacation-process evidence lacks maximum vacation duration. |
| ACAL-I03 | insufficient | near domain | Retention evidence lacks a number of years. |
| ACAL-I04 | insufficient | near domain | Coverage row lacks clinical specialties. |
| ACAL-I05 | insufficient | shared vocabulary | Remarcação é discutida, mas nenhum intervalo entre a primeira e a segunda é definido. |
| ACAL-I06 | insufficient | shared vocabulary | Benefits are discussed but none are named. |
| ACAL-I07 | insufficient | shared vocabulary | Audit/sharing areas are discussed but no people are named. |
| ACAL-I08 | insufficient | shared vocabulary | Plan rows lack annual consultation limits. |
| ACAL-I09 | insufficient | missing constraint | Business days do not establish guaranteed hours. |
| ACAL-I10 | insufficient | missing constraint | A 2026 record cannot establish status on the requested 2027 date. |
| ACAL-I11 | insufficient | missing constraint | Confirmation and a general area SLA do not establish a confirmation-specific maximum in hours. |
| ACAL-I12 | insufficient | missing constraint | The supplied rows do not contain the requested unit/plan combination. |
| ACAL-I13 | insufficient | plausible business | Directory entries lack budget and approver. |
| ACAL-I14 | insufficient | plausible business | Directory entries lack portal URL and 24x7 SLA. |
| ACAL-I15 | insufficient | plausible business | People-policy evidence lacks a travel-reimbursement process. |
| ACAL-I16 | insufficient | plausible business | Privacy directives lack vendors, contracts, and signatories. |

## Freeze and use

Any content change to the fixture requires a new version and fresh independent human review. Implementations may be calibrated against this development fixture, but results on it must not be represented as unseen holdout performance. The oracle evaluates the binary semantic boundary only; downstream answer generation and grounding remain separately testable boundaries.
