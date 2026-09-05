# Phase 5 — Schema Retrieval

**Complete.** All 453 test examples, paired against the frozen Phase 4 baseline
on the identical test set, identical model, identical decoding, identical
scoring code. The only variable is the schema shown to the model.

---

## Headline: retrieval halved the prompt and cost accuracy

| Metric | Baseline (full schema) | + Retrieval | Delta |
|---|---:|---:|---:|
| Strict execution accuracy | 10.82 % | 9.27 % | **−1.55** |
| Projection-tolerant accuracy | 45.92 % | 43.93 % | **−1.99** |
| Executable SQL | 98.90 % | 92.27 % | **−6.62** |
| Schema hallucination | 0.66 % | 3.31 % | **+2.65** |
| Syntax errors | 0.00 % | 0.00 % | 0.00 |
| Mean prompt tokens | 1,486 | 718 | **−768 (−52 %)** |

Retrieval delivered exactly the context reduction it promised — **52 % fewer
prompt tokens** — and made every accuracy metric worse. 13 examples the
baseline answered correctly became wrong; 6 went the other way.

**Ablation configuration 2 does not beat configuration 1 on this schema.**
Reported as measured.

### Sensitivity check

7 generations failed outright to provider throttling during the retrieval run —
an artefact of the free tier, not of retrieval. Excluding them:

| | Baseline | Retrieval | Delta |
|---|---:|---:|---:|
| Strict accuracy (n=446) | 9.87 % | 9.42 % | −0.45 |
| Executable SQL (n=446) | 98.88 % | 93.72 % | −5.16 |

The accuracy gap narrows to under half a point, but the executable-SQL gap
survives. The conclusion is unchanged: **retrieval did not help, and it broke
more SQL than it fixed.**

---

## The retriever itself performed well

Measured against `referenced_tables`, ground truth parsed from each gold query
back in Phase 3:

| | |
|---|---:|
| Table recall | 94.14 % |
| Complete retrieval (every needed table shown) | 90.07 % |
| Mean tables shown | 5.08 of 12 |
| Mean schema size | 2,407 chars vs 5,455 full (44 %) |

This is a competent retriever. The problem is not that it retrieved badly.

---

## Why it hurt — two distinct mechanisms

### 1. Missing tables cause hallucination (the predicted failure)

Hallucination rose 0.66 % → 3.31 %. Of the **14 new hallucinations**, **11 had
a missing table**. When retrieval drops a table the query needs, the model does
not refuse — it invents a column or a relationship to bridge the gap.

This is exactly the mechanism anticipated when the retriever was written:
*retrieval that drops a required table does not merely lose accuracy, it causes
the hallucination it was introduced to reduce.* It is now measured, not assumed.

Accuracy split by retrieval success is unambiguous:

| | Accuracy |
|---|---:|
| Retrieval complete (408 examples) | 10.29 % |
| Retrieval incomplete (45 examples) | **0.00 %** |

**Zero.** Not one of the 45 questions with a missing table was answered
correctly. A missing table is fatal, never merely unhelpful.

### 2. Extra tables cause ambiguous joins

`execution_error` rose 1 → 12, and the added failures are dominated by:

```
column reference "customer_id" is ambiguous
```

246 of the 453 test questions need exactly **one** table. For those, the
`expand=2` neighbour policy — chosen to maximise recall — pulls in everything
adjacent to the seed:

```
Q            : How many orders were placed each month of 2022?
gold tables  : orders
shown tables : customers, employees, order_items, orders,
               payments, products, shipments        (7 of 12)
```

Seven tables for a one-table question is barely a reduction from twelve, and
the adjacent tables invite joins that were never needed.

So the two mechanisms pull in opposite directions: **too few tables cause
hallucination, too many cause ambiguous joins.** The tuning that fixed the
first made the second worse.

---

## By difficulty

| Tier | n | Baseline | Retrieval | Change |
|---|---:|---:|---:|---:|
| easy | 126 | 0.0 % | 0.0 % | 0 |
| medium | 63 | 52.4 % | 44.4 % | **−5** |
| hard | 144 | 3.5 % | 2.1 % | −2 |
| very_hard | 36 | 0.0 % | 0.0 % | 0 |
| enterprise | 84 | 13.1 % | 13.1 % | 0 |

The damage concentrates in `medium` — single-table aggregations, precisely the
questions that need the fewest tables and therefore suffer most from
distractors.

## Outcome shifts

| Outcome | Baseline | Retrieval | Change |
|---|---:|---:|---:|
| correct | 49 | 42 | −7 |
| wrong_result | 399 | 376 | −23 |
| unknown_column | 3 | 14 | **+11** |
| execution_error | 1 | 12 | **+11** |
| unknown_table | 0 | 1 | +1 |
| generation_failed | 0 | 7 | +7 (throttle) |

---

## The structural reading

**This schema is 12 tables and 5,455 characters. The full DDL already fits
comfortably in the prompt.** Retrieval's benefits — shorter context, less
distraction, lower hallucination — have almost no room to appear, while its
costs land immediately.

Schema retrieval exists for databases with hundreds of tables, where the full
schema cannot fit at all and the real comparison is *retrieval vs nothing*, not
*retrieval vs full schema*. Measuring it at 12 tables measures the wrong regime,
and the honest conclusion is:

> **Retrieval is not beneficial at this schema size. That is a statement about
> the schema, not about retrieval.**

The 52 % token reduction is real and would matter at scale — it is the
mechanism by which retrieval pays off once a schema stops fitting. Here there
is nothing to pay for.

---

## Configuration, and what was wrong with it

Hyperparameters were tuned on the **validation split**, never on test.

| config (validation) | recall | complete | schema size |
|---|---:|---:|---:|
| k=6 expand=0 | 84.11 % | 85.40 % | 35.2 % |
| k=6 expand=1 | 93.25 % | 89.54 % | 49.4 % |
| **k=4 expand=2** (chosen) | **93.53 %** | **89.98 %** | **51.1 %** |
| k=6 expand=3 | 93.53 % | 89.98 % | 57.7 % |

Selected for the best recall at the smallest schema. **The selection criterion
was wrong.** It optimised recall alone, on the assumption that missing tables
cost more than extra ones. The results show both cost real accuracy, through
different mechanisms, and the optimum is a balance rather than a corner.

A retriever tuned on *end-to-end accuracy* on the validation split — rather
than on recall — would very likely land on a tighter configuration.

### Retriever design

Lexical, no embeddings, no new dependencies — deliberately, so an embedding
model has to earn its 8 GB-RAM cost against a measured baseline. Given that
lexical retrieval already achieves 94 % recall and *still* hurts accuracy,
embeddings would not have changed the conclusion.

- **Identifiers** — table and column names, tokenised and singularised
- **Documentation** — `COMMENT ON` text from Phase 2
- **Values** — distinct contents of 25 low-cardinality columns. The strongest
  signal: *"How many customers are from India?"* contains no schema identifier
  at all; only the literal `'India'` points at `customers.country`
- **Foreign-key path expansion** — connects chosen tables through the FK graph,
  so `customers` + `products` pulls in the `orders` / `order_items` bridge that
  no wording points at

---

## Recommended next steps

1. **Re-tune on end-to-end accuracy, not recall.** Sweep `expand` ∈ {0, 1, 2}
   against validation-split *execution accuracy*. One parameter, and the
   evidence says a tighter subset should win.
2. **Keep configuration 1 as the pipeline default** until retrieval beats it.
   The ablation study records the negative result; it does not adopt it.
3. **Do not delete this work.** The retriever, its recall harness, and the
   `referenced_tables` ground truth are all reusable, and Phase 14's ablation
   study needs this measurement to be honest.

---

## Reproducibility

```
model       Qwen/Qwen3-8B  rev b968826d9c46   fine_tuned=False  adapters=none
params      temperature=0.0, top_p=1.0, max_tokens=512, seed=20260808, /no_think
serving     HF Inference Providers, provider=featherless-ai
retriever   keyword_v1  k=4, expand=2, 25 value-indexed columns, embeddings=False
dataset     dataset/test/test.jsonl   ec7ddcae4f9d90d4   (unmodified)
database    PostgreSQL 18.1           data 5a018e291c3df4ad
DATA_AS_OF  2026-08-01T00:00:00+00:00
```

Harness self-test scores 100 % in both full-schema and retrieval modes, so
these differences are the system, not the evaluator. Latency is not compared:
271 of 453 retrieval calls were throttled by the free tier, making wall-clock
timings incomparable to the baseline run.

**Files:** `results.jsonl` (453) · `summary.json` · `generations.jsonl` (453 raw
outputs) · `retrieval_eval_k4e2.jsonl` (offline recall) ·
`retrieval_summary_k*.json` (validation tuning sweep).
