# Phase 4 — Base Model Benchmark

**Qwen/Qwen3-8B**, no fine-tuning, no schema retrieval, no SQL repair.

This is ablation configuration 1: the number every later configuration is
measured against. It is deliberately the weakest setup in the study — the point
is to establish an honest floor, not a good score.

---

## Headline

| Metric | Value |
|---|---:|
| Test examples | 453 |
| **Execution accuracy** | **0.00 %** |
| Correct | 0 |
| Executable SQL | 8 (1.77 %) |
| Ran but wrong answer | 8 (1.77 %) |
| Schema hallucination | 0.00 % |
| Mean generation latency | 1597 ms |
| Median generation latency | 1114 ms |
| Mean total latency | 34 ms |

Correctness is decided by **executing** both queries and comparing result
sets. SQL text is never compared — these two are different strings and the same
answer:

```sql
SELECT COUNT(*)           FROM customers WHERE country = 'India';
SELECT COUNT(customer_id) FROM customers WHERE country = 'India';
```

---

## Error rates

| Failure mode | Rate |
|---|---:|
| Ran, wrong answer | 1.77 % |
| Unknown column (hallucinated) | 0.00 % |
| Unknown table (hallucinated) | 0.00 % |
| Syntax error | 0.00 % |
| Other execution error | 0.00 % |
| Timeout | 0.00 % |
| No SQL produced | 0.00 % |
| Unparseable SQL | 0.00 % |
| Not read-only | 0.00 % |
| Generation failed | 98.23 % |

### Outcome distribution

Every example lands in exactly one bucket, so these sum to the total.

| Outcome | Count | Share | Meaning |
|---|---:|---:|---|
| `generation_failed` | 445 | 98.2 % | The inference API call failed after all retries. |
| `wrong_result` | 8 | 1.8 % | Ran successfully but returned a different result set than the gold query. The most dangerous failure: it looks like a valid answer. |

---

## Order-sensitive accuracy

The headline metric ignores row order. For questions like "the top 10 customers
by revenue", order is part of the answer, so those are also scored strictly.

| | |
|---|---:|
| Examples with `ORDER BY` in gold | 0 |
| Correct with order respected | 0 |
| Strict accuracy | 0.00 % |

A gap between this and the headline number means the model selected the right
rows but sorted them wrongly.

---

## Results by difficulty

| Difficulty | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `easy` | 126 | 0 | 0.0 % | 6.3 % |
| `enterprise` | 84 | 0 | 0.0 % | 0.0 % |
| `hard` | 144 | 0 | 0.0 % | 0.0 % |
| `medium` | 63 | 0 | 0.0 % | 0.0 % |
| `very_hard` | 36 | 0 | 0.0 % | 0.0 % |

## Results by domain

| Domain | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `catalogue` | 57 | 0 | 0.0 % | 0.0 % |
| `cross_domain` | 3 | 0 | 0.0 % | 0.0 % |
| `finance` | 66 | 0 | 0.0 % | 0.0 % |
| `hr` | 9 | 0 | 0.0 % | 0.0 % |
| `logistics` | 162 | 0 | 0.0 % | 0.0 % |
| `sales` | 156 | 0 | 0.0 % | 5.1 % |

## Results by SQL pattern

Query features are not exclusive — one query can be counted under several.

| SQL feature | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `avg` | 48 | 0 | 0.0 % | 0.0 % |
| `business_rule` | 72 | 0 | 0.0 % | 0.0 % |
| `churn` | 24 | 0 | 0.0 % | 0.0 % |
| `count` | 57 | 0 | 0.0 % | 0.0 % |
| `cross_domain` | 9 | 0 | 0.0 % | 0.0 % |
| `date` | 111 | 0 | 0.0 % | 7.2 % |
| `date_diff` | 6 | 0 | 0.0 % | 0.0 % |
| `group_by` | 174 | 0 | 0.0 % | 0.0 % |
| `having` | 48 | 0 | 0.0 % | 0.0 % |
| `join` | 48 | 0 | 0.0 % | 0.0 % |
| `limit` | 66 | 0 | 0.0 % | 0.0 % |
| `multi_join` | 99 | 0 | 0.0 % | 0.0 % |
| `not_exists` | 6 | 0 | 0.0 % | 0.0 % |
| `null_handling` | 42 | 0 | 0.0 % | 0.0 % |
| `order_by` | 78 | 0 | 0.0 % | 0.0 % |
| `partition` | 12 | 0 | 0.0 % | 0.0 % |
| `row_number` | 12 | 0 | 0.0 % | 0.0 % |
| `select` | 126 | 0 | 0.0 % | 6.3 % |
| `subquery` | 6 | 0 | 0.0 % | 0.0 % |
| `sum` | 69 | 0 | 0.0 % | 0.0 % |
| `where` | 162 | 0 | 0.0 % | 4.9 % |
| `window` | 18 | 0 | 0.0 % | 0.0 % |

---

## Failed examples

**bench-000322** · `e13` · easy / sales

> Show orders placed in 2023.

```sql
-- gold
SELECT order_id, customer_id, order_date, total_amount FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023

-- predicted
SELECT * FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023
```

`wrong_result` — gold returned 6783 rows, prediction returned 6783

**bench-000330** · `e13` · easy / sales

> Which orders were made during 2025?

```sql
-- gold
SELECT order_id, customer_id, order_date, total_amount FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2025

-- predicted
(nothing produced)
```

`generation_failed` — HfHubHTTPError: 402 Client Error: Payment Required for url: https://router.huggingface.co/nscale/v1/chat/completions (Request ID: Root=1-6a8d4977-018eeaf232746fa539cfb000;8698ee43-cf57-46ce-b8d9-e9357

**bench-000323** · `e13` · easy / sales

> List all orders from the year 2023.

```sql
-- gold
SELECT order_id, customer_id, order_date, total_amount FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023

-- predicted
SELECT * FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023
```

`wrong_result` — gold returned 6783 rows, prediction returned 6783

**bench-000324** · `e13` · easy / sales

> Which orders were made during 2023?

```sql
-- gold
SELECT order_id, customer_id, order_date, total_amount FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023

-- predicted
SELECT * FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023
```

`wrong_result` — gold returned 6783 rows, prediction returned 6783

**bench-000325** · `e13` · easy / sales

> Show orders placed in 2026.

```sql
-- gold
SELECT order_id, customer_id, order_date, total_amount FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2026

-- predicted
SELECT * FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2026
```

`wrong_result` — gold returned 12770 rows, prediction returned 12770

---

## Reproducibility

| | |
|---|---|
| Model | `Qwen/Qwen3-8B` |
| Model revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Serving | hf_inference_providers / auto |
| Fine-tuned | False |
| Adapters | none |
| Inference params | `{'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 512, 'seed': 20260808}` |
| Prompt version | `v1` (hash `8288e41a496531a9`) |
| Schema mode | `full_schema_no_retrieval` (hash `d03619e711661bc5`) |
| Dataset | `C:/Users/dell/Desktop/Enterprice-text to -SQL/dataset/test/test.jsonl` (hash `ec7ddcae4f9d90d4`) |
| Database | PostgreSQL 18.1 on x86_64-windows |
| Data fingerprint | `5a018e291c3df4ad` |
| DATA_AS_OF | `2026-08-01T00:00:00+00:00` |

Every one of these must match for a future run to be comparable. A changed
prompt hash means the next score measures prompt engineering; a changed data
fingerprint means it measures a different database.

### Gold fingerprint drift

Gold queries were re-executed and compared against the fingerprints recorded in Phase 3:

- RUN ABORTED: 7 unrecoverable generation errors (exhausted credits, bad token, or no model access). Results are partial and must not be reported as a baseline.

---

## What this number is not

This is the floor, not a verdict on the model. It has no schema retrieval, no
SQL repair, no few-shot examples, and no fine-tuning. Phases 6, 9 and 11 each
add one of those, and each is measured against this figure.

No claim that fine-tuning helps can be made until Phase 10 produces a
comparable number from an identical harness.
