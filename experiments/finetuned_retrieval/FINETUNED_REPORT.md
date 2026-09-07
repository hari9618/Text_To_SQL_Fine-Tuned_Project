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
| **Execution accuracy** | **41.72 %** |
| Correct | 189 |
| Executable SQL | 430 (94.92 %) |
| Ran but wrong answer | 241 (53.20 %) |
| Schema hallucination | 3.53 % |
| Mean generation latency | 2268 ms |
| Median generation latency | 2077 ms |
| SQL execution latency (median) | 16 ms |

Latency is measured over first-attempt calls only (0
of 453 calls were retried after provider throttling; counting
their backoff sleeps would report HuggingFace's rate limiter as model speed).

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
| Ran, wrong answer | 53.20 % |
| Unknown column (hallucinated) | 3.53 % |
| Unknown table (hallucinated) | 0.00 % |
| Syntax error | 0.00 % |
| Other execution error | 1.55 % |
| Timeout | 0.00 % |
| No SQL produced | 0.00 % |
| Unparseable SQL | 0.00 % |
| Not read-only | 0.00 % |
| Generation failed | 0.00 % |

### Outcome distribution

Every example lands in exactly one bucket, so these sum to the total.

| Outcome | Count | Share | Meaning |
|---|---:|---:|---|
| `wrong_result` | 241 | 53.2 % | Ran successfully but returned a different result set than the gold query. The most dangerous failure: it looks like a valid answer. |
| `correct` | 189 | 41.7 % | Correct — result matched the gold answer. |
| `unknown_column` | 16 | 3.5 % | Referenced a column that does not exist — schema hallucination. |
| `execution_error` | 7 | 1.5 % | Executed but raised another database error (type mismatch, ambiguous reference, grouping violation). |

---

## The headline number is an undercount — here is why

Many benchmark questions never say *which columns* to return. "Show orders
placed in 2023" is answered just as well by `SELECT *` as by the four columns
the gold query happens to name. Strict comparison marks those wrong even when
the rows are byte-identical.

| | Count | Share |
|---|---:|---:|
| Correct (exact match) | 189 | 41.72 % |
| Right rows, different columns | 0 | 0.00 % |
| Genuinely wrong rows | 241 | 53.20 % |
| **Projection-tolerant accuracy** | **189** | **41.72 %** |

Both numbers are real and both matter:

* **41.72 %** is the honest strict floor, and the
  figure Phase 10 must beat on identical terms.
* **41.72 %** is how often the model
  found the right *rows*.

The gap between them is the share of the benchmark that tests column
convention rather than SQL reasoning. It is reported separately so that a
fine-tuned model cannot post a large apparent gain merely by learning which
columns this dataset likes to select.

---

## Order-sensitive accuracy

The headline metric ignores row order. For questions like "the top 10 customers
by revenue", order is part of the answer, so those are also scored strictly.

| | |
|---|---:|
| Examples with `ORDER BY` in gold | 253 |
| Correct with order respected | 102 |
| Strict accuracy | 40.32 % |

A gap between this and the headline number means the model selected the right
rows but sorted them wrongly.

---

## Results by difficulty

| Difficulty | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `hard` | 144 | 92 | 63.9 % | 92.4 % |
| `medium` | 63 | 38 | 60.3 % | 100.0 % |
| `enterprise` | 84 | 33 | 39.3 % | 94.0 % |
| `easy` | 126 | 24 | 19.1 % | 100.0 % |
| `very_hard` | 36 | 2 | 5.6 % | 80.6 % |

## Results by domain

| Domain | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `catalogue` | 57 | 34 | 59.6 % | 100.0 % |
| `hr` | 9 | 5 | 55.6 % | 100.0 % |
| `logistics` | 162 | 69 | 42.6 % | 88.3 % |
| `finance` | 66 | 26 | 39.4 % | 95.5 % |
| `sales` | 156 | 55 | 35.3 % | 100.0 % |
| `cross_domain` | 3 | 0 | 0.0 % | 66.7 % |

## Results by SQL pattern

Query features are not exclusive — one query can be counted under several.

| SQL feature | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `having` | 48 | 42 | 87.5 % | 100.0 % |
| `avg` | 48 | 41 | 85.4 % | 87.5 % |
| `join` | 48 | 33 | 68.8 % | 89.6 % |
| `null_handling` | 42 | 27 | 64.3 % | 100.0 % |
| `multi_join` | 99 | 54 | 54.5 % | 88.9 % |
| `group_by` | 174 | 94 | 54.0 % | 92.0 % |
| `count` | 57 | 30 | 52.6 % | 100.0 % |
| `subquery` | 6 | 3 | 50.0 % | 100.0 % |
| `business_rule` | 72 | 33 | 45.8 % | 93.1 % |
| `order_by` | 78 | 33 | 42.3 % | 85.9 % |
| `sum` | 69 | 26 | 37.7 % | 84.1 % |
| `where` | 162 | 58 | 35.8 % | 100.0 % |
| `limit` | 66 | 23 | 34.9 % | 83.3 % |
| `select` | 126 | 24 | 19.1 % | 100.0 % |
| `churn` | 24 | 3 | 12.5 % | 100.0 % |
| `date` | 111 | 10 | 9.0 % | 100.0 % |
| `window` | 18 | 1 | 5.6 % | 100.0 % |
| `cross_domain` | 9 | 0 | 0.0 % | 44.4 % |
| `date_diff` | 6 | 0 | 0.0 % | 0.0 % |
| `not_exists` | 6 | 0 | 0.0 % | 100.0 % |
| `partition` | 12 | 0 | 0.0 % | 100.0 % |
| `row_number` | 12 | 0 | 0.0 % | 100.0 % |

---

## Failure patterns

Classification of the 241 genuinely wrong answers, excluding the
column-projection artefact above.

| Pattern | Count | Share |
|---|---:|---:|
| Right row count, different values (usually a different metric definition) | 179 | 74 % |
| LEFT JOIN where gold used INNER (kept unmatched rows) | 18 | 7 % |
| Returned zero rows (over-filtered) | 17 | 7 % |
| Too many rows (under-filtered) | 14 | 6 % |
| Too few rows (over-filtered) | 7 | 3 % |
| INNER JOIN where gold used LEFT (dropped rows) | 3 | 1 % |
| Invented a LIMIT the question never asked for | 2 | 1 % |
| Added DISTINCT, collapsing legitimate duplicates | 1 | 0 % |

The largest bucket is *not* broken SQL. It is the model computing a defensible
but different quantity — most often revenue summed from `order_items` lines
when the gold query used the denormalised `orders.total_amount`. Both are
reasonable readings of "revenue"; the schema supports two answers and the
question does not disambiguate. That is a business-rule gap, which is exactly
what fine-tuning on this domain is expected to close.

---

## Successful examples

**bench-000652** · `m04` · medium / sales

> How many orders are there in each status?

```sql
-- gold
SELECT status, COUNT(*) AS order_count FROM orders GROUP BY status ORDER BY order_count DESC

-- predicted
SELECT status, COUNT(*) AS order_count FROM orders GROUP BY status ORDER BY order_count DESC
```

**bench-000653** · `m04` · medium / sales

> Count orders by status.

```sql
-- gold
SELECT status, COUNT(*) AS order_count FROM orders GROUP BY status ORDER BY order_count DESC

-- predicted
SELECT status, COUNT(*) AS order_count FROM orders GROUP BY status ORDER BY order_count DESC
```

**bench-000654** · `m04` · medium / sales

> Show the number of orders per status.

```sql
-- gold
SELECT status, COUNT(*) AS order_count FROM orders GROUP BY status ORDER BY order_count DESC

-- predicted
SELECT status, COUNT(*) AS order_count FROM orders GROUP BY status ORDER BY order_count DESC
```

**bench-000664** · `m08` · medium / sales

> What is the total revenue for each year?

```sql
-- gold
SELECT EXTRACT(YEAR FROM order_date) AS year, SUM(total_amount) AS revenue FROM orders GROUP BY year ORDER BY year

-- predicted
SELECT EXTRACT(YEAR FROM order_date) AS year, SUM(total_amount) AS revenue FROM orders GROUP BY year ORDER BY year
```

---

## Failed examples

**bench-000322** · `e13` · easy / sales

> Show orders placed in 2023.

```sql
-- gold
SELECT order_id, customer_id, order_date, total_amount FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023

-- predicted
SELECT order_id, customer_id, order_date FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023
```

`wrong_result` — gold returned 6783 rows, prediction returned 6783

**bench-001717** · `v08` · very_hard / logistics

> Which carrier delivers fastest on average?

```sql
-- gold
SELECT s.carrier, AVG(EXTRACT(EPOCH FROM (s.delivered_date - s.shipped_date)) / 86400.0) AS avg_days FROM shipments s WHERE s.delivered_date IS NOT NULL AND s.shipped_date IS NOT NULL GROUP BY s.carrier ORDER BY avg_days ASC

-- predicted
SELECT c.carrier, AVG(DATE_PART('day', s.delivered_date - s.shipped_date)) AS average_days FROM shipments s JOIN orders o ON o.order_id = s.order_id JOIN customers c ON c.customer_id = o.customer_id WHERE s.delivered_date IS NOT NULL GROUP BY c.carrier ORDER BY average_days
```

`unknown_column` — column c.carrier does not exist

**bench-001718** · `v08` · very_hard / logistics

> Show mean delivery days per carrier.

```sql
-- gold
SELECT s.carrier, AVG(EXTRACT(EPOCH FROM (s.delivered_date - s.shipped_date)) / 86400.0) AS avg_days FROM shipments s WHERE s.delivered_date IS NOT NULL AND s.shipped_date IS NOT NULL GROUP BY s.carrier ORDER BY avg_days ASC

-- predicted
SELECT carrier, AVG(DATEDIFF(delivered_date, shipped_date)) AS average_delivery_days FROM shipments WHERE delivered_date IS NOT NULL GROUP BY carrier ORDER BY average_delivery_days DESC
```

`execution_error` — function datediff(timestamp with time zone, timestamp with time zone) does not exist

**bench-000323** · `e13` · easy / sales

> List all orders from the year 2023.

```sql
-- gold
SELECT order_id, customer_id, order_date, total_amount FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023

-- predicted
SELECT order_id, customer_id, order_date FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023
```

`wrong_result` — gold returned 6783 rows, prediction returned 6783

**bench-000324** · `e13` · easy / sales

> Which orders were made during 2023?

```sql
-- gold
SELECT order_id, customer_id, order_date, total_amount FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023

-- predicted
SELECT order_id, customer_id, order_date FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023
```

`wrong_result` — gold returned 6783 rows, prediction returned 6783

**bench-000325** · `e13` · easy / sales

> Show orders placed in 2026.

```sql
-- gold
SELECT order_id, customer_id, order_date, total_amount FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2026

-- predicted
SELECT order_id, customer_id, order_date FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2026
```

`wrong_result` — gold returned 12770 rows, prediction returned 12770

---

## Reproducibility

| | |
|---|---|
| Model | `Qwen/Qwen3-8B` |
| Model revision | `b968826d9c46` |
| Serving | qlora_adapter_replayed / kaggle-t4-local-generation |
| Fine-tuned | True |
| Adapters | [{'path': 'models/finetuned/final_adapter', 'method': 'qlora', 'r': 16, 'alpha': 32, 'dropout': 0.05, 'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], 'trainable_params': 43646976, 'epochs': 1.0, 'train_on_completion_only': True}] |
| Inference params | `None` |
| Prompt version | `v1` (hash `8288e41a496531a9`) |
| Schema mode | `retrieved_keyword_v1` (hash `d03619e711661bc5`) |
| Dataset | `C:/Users/dell/Desktop/Enterprice-text to -SQL/dataset/test/test.jsonl` (hash `ec7ddcae4f9d90d4`) |
| Database | PostgreSQL 18.1 on x86_64-windows |
| Data fingerprint | `5a018e291c3df4ad` |
| DATA_AS_OF | `2026-08-01T00:00:00+00:00` |

Every one of these must match for a future run to be comparable. A changed
prompt hash means the next score measures prompt engineering; a changed data
fingerprint means it measures a different database.

### Gold fingerprint check

All gold queries were re-executed and reproduced their Phase 3 fingerprints exactly — the database has not drifted since validation.

---

## What this number is not

This is the floor, not a verdict on the model. It has no schema retrieval, no
SQL repair, no few-shot examples, and no fine-tuning. Phases 6, 9 and 11 each
add one of those, and each is measured against this figure.

No claim that fine-tuning helps can be made until Phase 10 produces a
comparable number from an identical harness.
