# Evaluation and reliability

## Inspection and scope

The Phase 5 baseline had 209 passing deterministic tests. Inspection covered ingestion adapters, normalization and atomic storage, JSONL contracts, indexing/ranking/filtering/pagination, provider configuration, native tool schemas, dispatch, citation validation, orchestration, state, limits, logging, and existing fixtures.

Existing tests already covered corrupt input, workbook safety, missing formula caches, citations, provider authentication/timeouts/rate limits, unknown tools, invalid arguments, missing records, incomplete indexes, evidence budgets, repeated calls, interrupted execution, session isolation, and context/iteration/tool/time limits. Phase 5 keeps these tests and adds evaluation-specific checks instead of duplicating them.

Gaps addressed: independent numerical and answer grading, citation support beyond record existence, reproducible datasets/reports, sanitized evaluation traces, latency measurement, category breakdowns, regression comparison, and deterministic CI.

Fault injection confirmed that malformed tool JSON or unexpected result shapes crashed the agent. A regression test now covers seven malformed envelopes. The shared agent loop stops with `failed / invalid_response`, counts the attempted tool call, preserves collected evidence, avoids exposing the malformed payload, and makes no follow-up model request. Existing provider and tool behavior remains intact.

Pre-existing ingestion comment edits were retained and formatted so Ruff passes. No document-processing feature, retrieval engine, production calculator, persistent memory, hosted service, or new dependency was added.

## Fixtures and ground truth

`evals/corpus/boq.jsonl` and `specification.jsonl` contain **19 explicitly synthetic records**: two document roots, twelve BOQ entries, and five specification paragraphs. Their SHA-like document IDs are fixture identities, not claims that unpublished source files exist. Every fixture record is marked synthetic. No private construction document or derived real-project record is included.

The BOQ includes foundation/column concrete, two excavation rows, rebar, formwork, a negative credit, a zero quantity, decimal grout, an existing subtotal, an overlapping duplicate, and an alternative revision in EUR. Specifications include strength, curing, cover, conflicting draft requirements, and malicious document instructions.

`evals/datasets/golden.jsonl` contains **54 version-1 cases**:

- 14 retrieval cases: exact identifiers, leading zeros, filters, PDF pages, absent terms, and exhaustive pagination.
- 40 agent cases: quantities/rates/amounts, min/max, subtotals/totals, units and currencies, overlap and subtotal exclusions, PDF facts, cross-document synthesis, complete-record inspection, discovery, conversation follow-ups/switches/clear, ambiguity, missing information, conflicts, malicious documents/users, and injected reliability failures.

Every case explicitly declares `synthetic` and `verified`. Unverified cases are skipped and excluded from accuracy denominators. Before execution, verified labels are checked against the corpus: referenced records must exist, scalar fields must match, PDF fact text must contain the reference value and unit, and Decimal calculations must match numerical labels exactly. Invalid ground truth aborts the run rather than being silently adjusted. Quantities, currencies, and rate units must agree with the source fixtures.

Agent responses never generate the golden labels. `scripts.json` is a separate set of fixed offline provider transcripts; the offline provider does not read expected answers. Mutation tests deliberately change those transcripts to verify rejection of incorrect numbers, units, contradictory duplicated facts, unsupported prose, and valid-but-wrong citations.

## Dataset schema

The schema lives in `evals/schemas.py`. Important case fields:

| Field | Meaning |
| --- | --- |
| `id`, `version`, `category`, `layer` | Stable identity and retrieval/agent layer |
| `synthetic`, `verified`, `live_eligible` | Provenance, label status, and whether injected scripts apply live |
| `question`, `query`, `filters`, `k` | Question and independent retrieval request |
| `expected_ids`, `exact_item`, `exhaustive` | Evidence labels and applicable retrieval metrics |
| `facts` | Independent, labelled objective expectations |
| `should_abstain`, `expected_status`, `expected_stop` | Required uncertainty/failure behavior |
| `required_tools`, `max_tools` | Required capabilities and efficiency ceiling, without fixing call order |
| `setup_questions` | Isolated conversation setup; `null` clears the session |
| `config` | Application execution budgets for this case |

A fact names a label, reference value, unit, whether it is numerical, supporting record IDs, a source field path, an operation (`value`, `contains`, `sum`, `min`, `max`), and an optional absolute Decimal tolerance. Tolerance defaults to zero. Add one only with documented reference rounding. Item identifiers are text, so `001` differs from `1`.

Controlled objective questions request one `label: value unit` per line. This makes automated checks explicit and conservative. Arbitrary explanatory prose needs manual semantic review; no subjective LLM judge is configured. Abstention grading checks structured status/type and evidence requirements, not the semantic quality of its explanation.

## Commands

```bash
# Entire offline evaluation: no API key, no network client.
uv run python main.py eval --mode offline

# Select a case or category.
uv run python -m evals.runner --case decimal_precision --output-dir reports/precision
uv run python main.py eval --category conversation

# Keep a baseline, then compare identical data after a code/prompt change.
uv run python main.py eval --output-dir reports/baseline
uv run python main.py eval --baseline reports/baseline/latest.json

# Custom independently labelled cases and JSONL corpus.
uv run python main.py eval --dataset path/to/verified.jsonl --corpus path/to/processed
```

Custom offline datasets also need independent fixed transcripts supplied with `--scripts path/to/scripts.json`. The dataset/corpus/scripts hashes, selected case IDs, mode, report schema version, UTC timestamp, and live model name identify a run. Comparison rejects mismatched datasets/corpora/transcripts/selections/modes and identifies per-check regressions even when aggregate pass rates are unchanged. Copy the baseline before writing a new run into the same output directory.

Live execution is implemented but has **not** been run as part of Phase 5. It requires both explicit flags and provider credentials:

```bash
export OPENAI_API_KEY=...  # Supply securely; do not commit credentials.
export OPENAI_MODEL=...   # A model supported by the existing application.
uv run python main.py eval --mode live --allow-paid --output-dir reports/live
```

This command incurs API charges. Ordinary pytest and CI never run it. The default live corpus remains synthetic. Live cases replay setup questions with fresh isolated sessions and invoke the existing LLM/client and agent. Six offline-only fault-injection cases are skipped live; those behaviors remain tested deterministically. Provider-reported usage is recorded only when available for every request in the scored turn. Offline token/cost fields are null. No cost estimator or model pricing table is implemented. Repeat authorized live runs to assess variability; do not compare their accuracy as if it were deterministic.

Exit codes: `0` means all executed checks passed, `1` means a case or baseline regression failed, and `2` means setup/schema/opt-in/configuration failed. Skips and measured denominators are explicit in reports.

## Metrics

| Metric | Definition and scope |
| --- | --- |
| Precision@K | Unique relevant hits among top K / K; unfilled positions count against precision |
| Recall@K | Relevant hits in top K / all labelled relevant records; undefined for empty labels |
| MRR | Mean reciprocal rank of the first relevant result within K; undefined for empty labels |
| Exact item accuracy | Top-ranked record is expected, on exact-item cases only |
| Exhaustive completeness | All known relevant records recovered across actual pagination; duplicates fail the case |
| Answer accuracy | All objective labelled facts and output-format/status checks pass; abstention-only/fault cases excluded |
| Numerical accuracy | Mean per-case proportion of labelled numerical values within explicit tolerance, separately from units |
| Unit correctness | Mean per-case unit/currency correctness; `m³` and `m3` normalize, USD and EUR do not |
| Item selection correctness | Required supporting items and fields were actually observed and cited, separately from the numerical answer |
| Unsupported/incorrect numerical claims | Incorrect or unsupported numerical claims / labelled numerical claims |
| Citation validity | Real, exact, available source references / all emitted citations |
| Citation support | Correct labelled claims with cited, actually observed supporting fields / evaluated claims, including ungraded extra lines |
| Grounded answer rate | Every labelled claim is correct, supported and cited, format is valid, and expected evidence is complete; factual cases only |
| Complete evidence rate | Required record IDs were observed; cases with no required evidence excluded |
| Unsupported claim rate | Unsupported evaluated or extra claims / evaluated claims |
| Tool correctness | Required capabilities and call ceiling satisfied; exhaustive cases require an observed complete result, not just an attempted request |

Retrieval metrics are macro-averaged over applicable cases. Valid citation IDs alone do not establish support: graders check the correct reference, expected supporting item, field presence in the actual evidence supplied, field agreement with the independently labelled corpus, answer value, and unit. Missing citations, fabricated locations, and wrong supporting records fail separately.

Formatting normalization uses Unicode NFKC, case folding, and whitespace normalization. Numerical parsing uses Decimal and accepts unambiguous thousands separators. It rejects decimal commas, NaN, wrong signs, larger numbers containing the expected digits, wrong units/currencies, duplicate labelled assertions, and extra unsupported text. It does not silently round answers into correctness.

Reports include retrieval-only and end-to-end turn latency, individual provider-request latency, tool execution latency, request/iteration/tool counts, and live provider token counts where available. Median is reported for any nonempty sample; p95 uses nearest rank and requires at least 20 samples. Index loading and conversation setup are excluded from scored-turn timing and counts. Tool duration includes retrieval plus validation/serialization; it is not presented as an isolated internal search benchmark. Offline provider latency measures local script execution, not API latency.

## Reports, debugging, and privacy

`latest.json` contains each case, applicable metrics, boolean checks, failure taxonomy, expected/actual behavior summaries, sanitized observable events, available/cited record IDs, source locations for synthetic cases, and a recommended debugging area. Synthetic questions and reference facts are included; non-synthetic question text and source-location details are omitted. Provider answer text, document contents, full arguments, secrets, and private reasoning are never serialized by the reporter. Case IDs/labels/record IDs should themselves be non-sensitive. Review custom datasets before sharing reports.

`latest.md` summarizes totals, categories, metrics, failures, limitations, and optional baseline comparison. Failure classes include retrieval miss, wrong record, incomplete evidence, numerical/unit errors, citation/support errors, tool selection, unauthorized execution, context, timeout, API/limit failures, and invalid output. No production module depends on graders.

For a failed case, first inspect its checks and `debug_area`. A missing expected evidence ID indicates retrieval/selection/incompleteness; correct evidence with a wrong numerical value indicates answer reasoning; wrong citation metadata indicates provenance; repeated/failed requests indicate tool or orchestration behavior. The dataset supplies the original question and labels if report content is redacted.

## Acceptance and remaining blockers

### OpenRouter runtime regression

A subsequent live debugging run with the configured free model reproduced repeated tool errors on the question `tell me about this file`. Invalid record kinds, string-encoded page filters, and omitted optional search controls exhausted the recoverable-error budget. The strict tool generator had also made every optional nested filter mandatory in the advertised schema.

Search controls now have safe defaults, record kinds use the same enum as ingestion, and OpenRouter receives schemas that preserve optional controls. The tool boundary normalizes decimal page strings and null-like strings only for nullable controls that cannot legitimately have those string values; named worksheets such as `None` remain unchanged. The direct retrieval API remains strict. Invalid types, unknown fields, invalid kinds, nonpositive pages, and out-of-range limits are still rejected. Correction feedback lists safe field names and constraints without echoing private argument values.

When retrieval exhausts its evidence budget, the agent makes a final-answer attempt using available evidence within its existing request/time/context limits. Remaining pending calls receive budget-error outputs without executing more retrieval. The final request advertises no tools and instructs the model to acknowledge evidence limitations. The application rejects any further tool request. Source display groups identical locations within a document; JSON keeps every record citation.

These changes have deterministic regression tests and preserve the numerical/source safeguards. Ad hoc debugging requests are not benchmark measurements or proof that this model always follows schemas or produces correct summaries.

The OpenRouter client also includes the answer schema in its instructions, supports `OPENAI_REASONING_EFFORT`, and reserves the last allowed request for an answer. The configured MiniMax model advertises mandatory reasoning with default effort `max`; the local `.env` was tuned to `OPENAI_REASONING_EFFORT=low` and `OPENAI_MAX_OUTPUT_TOKENS=8192`. See [OpenRouter's reasoning-budget documentation](https://github.com/OpenRouterTeam/docs/blob/main/guides/best-practices/reasoning-tokens.mdx). Invalid citation metadata gets at most one correction attempt for OpenRouter, using exact retrieved citation options; validation still rejects unsupported citations and all original execution limits apply.

After these fixes, a live smoke test of the exact question `tell me about this file` completed with the configured free model: 4 model requests, 5 tool calls, a 1,935-character answer, and 4 retrieved-source citations. No paid model route was used. This single successful execution checks runtime integration, not semantic answer accuracy or general model reliability. The expanded deterministic suite passed **267 tests**, all **54 offline evaluation cases** passed, and the existing baseline comparison found no per-check regressions.

CI requires all deterministic tests and every verified offline case to pass. In particular, fabricated provenance, unauthorized tool execution, leaked test secrets, invalid final output acceptance, broken session isolation, and violated limits are blockers. Deliberate fault scenarios pass only when the expected controlled failure occurs. Positive mock transcripts do not demonstrate semantic attack resistance; boundary-enforcement tests independently verify that model text cannot invoke shell/Python or bypass argument schemas.

Use the first measured run as the baseline for these versioned synthetic fixtures. Track category and per-check regressions, especially numerical/citation/security checks; do not hide them behind an average. Retrieval P@K/recall are measurements rather than arbitrary claims that all real-world queries must reach 100%. Known lexical synonym gaps remain in the original Phase 2 evaluation.

Outstanding validation before real deployment: authorized live model runs, representative verified real-document labels, manual review of explanations/abstentions, OCR/layout limitations, reliable formula caches, lexical vocabulary coverage, document-scope completeness, and model arithmetic. This work establishes reproducible engineering checks; it does not certify production readiness.

## Measured Phase 5 baseline

This local run used only the committed synthetic corpus and fixed offline provider transcripts. The deterministic suite passed **243 tests**. Ruff checks and formatting passed. All **54 offline cases passed** (14 retrieval, 40 agent); no cases were skipped. Repeating the run against the same dataset/corpus found no per-check regressions. The GitHub workflow is configured; execution on hosted GitHub runners has not been observed.

| Measurement | Actual offline result |
| --- | --- |
| Precision@K | 0.928571 (14 retrieval cases, including one expected-empty query) |
| Recall@K | 0.961538 (13 cases with relevant labels) |
| MRR | 1.000000 |
| Exact item retrieval | 6/6 |
| Exhaustive retrieval completeness | 100% on the exhaustive retrieval case |
| Objective scripted answers | 29/29 |
| Labelled numerical values and units | 100%; 28 numerical claims |
| Valid citations | 38/38 |
| Supported labelled claims | 31/31 |
| Unsupported/incorrect numerical claims | 0/28 |
| Scripted tool behavior | 100% |
| Scored requests / tool executions | 79 / 49 |
| Retrieval-only median latency | 0.114 ms (14 samples; p95 unavailable) |
| Offline agent median / p95 latency | 2.882 / 5.148 ms (40 samples) |
| Live accuracy, API latency, token usage, costs | Not measured |

The exhaustive retrieval case uses K=1 with two relevant rows. Its first-page recall is 0.5, while actual pagination recovers both rows; these are different measurements. Positive scripted answers are expected to pass and establish reproducible contracts, not model intelligence. Negative mutation tests and injected faults verify failure detection. Local timing is machine-specific and does not predict paid API latency.

Dataset SHA-256: `be0aa7225edd8e241a1513b909715381bb9b1a0c0845bbb6d576a907ffb170ef`. Corpus SHA-256: `d39646e6331123cd0ebd7191a817f32439063b489f2260d8338a9509dcb362e6`. Full measured reports remain in `reports/latest.json` and `reports/latest.md`; use `reports/baseline/latest.json` for the local preserved comparison.
