# Official Frozen Baseline — Phase 4

**Approved and frozen.** These figures are the reference every later
configuration in the ablation study is measured against. They must not be
recomputed, re-tuned, or silently superseded. If the model, prompt, schema
rendering, dataset or database changes, the comparison is void and the baseline
must be re-run deliberately — not adjusted.

## Configuration 1 — base model, no retrieval, no repair

| Metric | Value |
|---|---:|
| Strict execution accuracy | **10.82 %** (49/453) |
| Projection-tolerant accuracy | **45.92 %** (208/453) |
| Executable SQL | **98.90 %** |
| Schema hallucination | **0.66 %** |
| Order-sensitive accuracy | **5.20 %** (14/269) |

Supporting figures: 0.00 % syntax errors; generation latency mean 1,330 ms /
median 1,233 ms (first-attempt calls only); 1,486 prompt + 57 completion tokens
per example.

## Frozen run identity

```
model       Qwen/Qwen3-8B   revision b968826d9c46
            fine_tuned=False, adapters=none
params      temperature=0.0, top_p=1.0, max_tokens=512, seed=20260808
            Qwen3 thinking mode disabled (/no_think)
prompt      v1                        8288e41a496531a9
schema      full_schema_no_retrieval  d03619e711661bc5
dataset     dataset/test/test.jsonl   ec7ddcae4f9d90d4
database    PostgreSQL 18.1           data 5a018e291c3df4ad
DATA_AS_OF  2026-08-01T00:00:00+00:00
```

## Why two accuracy figures are frozen, not one

35.10 % of the test set (159 examples) returns the **correct rows** under a
different column projection — 123 of them plain `SELECT *` — because many
questions never state which columns to return.

The test set is deliberately **not** modified to remove this ambiguity. Instead
both numbers are frozen:

* **Strict (10.82 %)** — the headline floor. Later phases must beat this on
  identical terms.
* **Projection-tolerant (45.92 %)** — how often the right *rows* were found.

Tracking only the strict figure would let a later configuration post a large
apparent gain purely by learning this dataset's column conventions. Tracking
only the tolerant figure would hide real regressions. Both move together, or
the change is an artefact.

## Evidence

`BASELINE_REPORT.md` (full analysis) · `results.jsonl` (453 per-example rows) ·
`summary.json` (metrics + metadata) · `generations.jsonl` (raw model outputs).

Harness self-test: gold SQL through the identical scoring path scores 100 %
(453/453), so this figure measures the model, not the evaluator.
