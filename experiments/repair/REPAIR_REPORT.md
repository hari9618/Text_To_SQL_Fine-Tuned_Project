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
| **Execution accuracy** | **52.10 %** |
| Correct | 236 |
| Executable SQL | 445 (98.23 %) |
| Ran but wrong answer | 209 (46.14 %) |
| Schema hallucination | 0.66 % |
| Mean generation latency | 3924 ms |
| Median generation latency | 3428 ms |
| SQL execution latency (median) | 12 ms |

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
| Ran, wrong answer | 46.14 % |
| Unknown column (hallucinated) | 0.66 % |
| Unknown table (hallucinated) | 0.00 % |
| Syntax error | 0.22 % |
| Other execution error | 0.88 % |
| Timeout | 0.00 % |
| No SQL produced | 0.00 % |
| Unparseable SQL | 0.00 % |
| Not read-only | 0.00 % |
| Generation failed | 0.00 % |

### Outcome distribution

Every example lands in exactly one bucket, so these sum to the total.

| Outcome | Count | Share | Meaning |
|---|---:|---:|---|
| `correct` | 236 | 52.1 % | Correct — result matched the gold answer. |
| `wrong_result` | 209 | 46.1 % | Ran successfully but returned a different result set than the gold query. The most dangerous failure: it looks like a valid answer. |
| `execution_error` | 4 | 0.9 % | Executed but raised another database error (type mismatch, ambiguous reference, grouping violation). |
| `unknown_column` | 3 | 0.7 % | Referenced a column that does not exist — schema hallucination. |
| `syntax_error` | 1 | 0.2 % | PostgreSQL rejected the SQL as malformed. |

---

## The headline number is an undercount — here is why

Many benchmark questions never say *which columns* to return. "Show orders
placed in 2023" is answered just as well by `SELECT *` as by the four columns
the gold query happens to name. Strict comparison marks those wrong even when
the rows are byte-identical.

| | Count | Share |
|---|---:|---:|
| Correct (exact match) | 236 | 52.10 % |
| Right rows, different columns | 0 | 0.00 % |
| Genuinely wrong rows | 209 | 46.14 % |
| **Projection-tolerant accuracy** | **236** | **52.10 %** |

Both numbers are real and both matter:

* **52.10 %** is the honest strict floor, and the
  figure Phase 10 must beat on identical terms.
* **52.10 %** is how often the model
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
| Examples with `ORDER BY` in gold | 267 |
| Correct with order respected | 142 |
| Strict accuracy | 53.18 % |

A gap between this and the headline number means the model selected the right
rows but sorted them wrongly.

---

## Results by difficulty

| Difficulty | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `hard` | 144 | 122 | 84.7 % | 100.0 % |
| `medium` | 63 | 38 | 60.3 % | 100.0 % |
| `enterprise` | 84 | 44 | 52.4 % | 95.2 % |
| `easy` | 126 | 29 | 23.0 % | 100.0 % |
| `very_hard` | 36 | 3 | 8.3 % | 88.9 % |

## Results by domain

| Domain | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `hr` | 9 | 7 | 77.8 % | 100.0 % |
| `catalogue` | 57 | 33 | 57.9 % | 100.0 % |
| `logistics` | 162 | 92 | 56.8 % | 96.9 % |
| `sales` | 156 | 75 | 48.1 % | 100.0 % |
| `finance` | 66 | 29 | 43.9 % | 97.0 % |
| `cross_domain` | 3 | 0 | 0.0 % | 66.7 % |

## Results by SQL pattern

Query features are not exclusive — one query can be counted under several.

| SQL feature | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `having` | 48 | 42 | 87.5 % | 100.0 % |
| `multi_join` | 99 | 86 | 86.9 % | 100.0 % |
| `avg` | 48 | 40 | 83.3 % | 93.8 % |
| `sum` | 69 | 56 | 81.2 % | 100.0 % |
| `limit` | 66 | 53 | 80.3 % | 100.0 % |
| `order_by` | 78 | 61 | 78.2 % | 100.0 % |
| `group_by` | 174 | 123 | 70.7 % | 99.4 % |
| `churn` | 24 | 16 | 66.7 % | 100.0 % |
| `join` | 48 | 32 | 66.7 % | 93.8 % |
| `null_handling` | 42 | 27 | 64.3 % | 97.6 % |
| `business_rule` | 72 | 44 | 61.1 % | 95.8 % |
| `count` | 57 | 29 | 50.9 % | 100.0 % |
| `where` | 162 | 63 | 38.9 % | 100.0 % |
| `subquery` | 6 | 2 | 33.3 % | 100.0 % |
| `select` | 126 | 29 | 23.0 % | 100.0 % |
| `date` | 111 | 23 | 20.7 % | 99.1 % |
| `window` | 18 | 1 | 5.6 % | 100.0 % |
| `cross_domain` | 9 | 0 | 0.0 % | 77.8 % |
| `date_diff` | 6 | 0 | 0.0 % | 50.0 % |
| `not_exists` | 6 | 0 | 0.0 % | 83.3 % |
| `partition` | 12 | 0 | 0.0 % | 100.0 % |
| `row_number` | 12 | 0 | 0.0 % | 100.0 % |

---

## Failure patterns

Classification of the 209 genuinely wrong answers, excluding the
column-projection artefact above.

| Pattern | Count | Share |
|---|---:|---:|
| Right row count, different values (usually a different metric definition) | 164 | 78 % |
| Returned zero rows (over-filtered) | 15 | 7 % |
| Too many rows (under-filtered) | 15 | 7 % |
| LEFT JOIN where gold used INNER (kept unmatched rows) | 5 | 2 % |
| INNER JOIN where gold used LEFT (dropped rows) | 5 | 2 % |
| Invented a LIMIT the question never asked for | 3 | 1 % |
| Too few rows (over-filtered) | 2 | 1 % |

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

**bench-001719** · `v08` · very_hard / logistics

> Compare carriers by average delivery time.

```sql
-- gold
SELECT s.carrier, AVG(EXTRACT(EPOCH FROM (s.delivered_date - s.shipped_date)) / 86400.0) AS avg_days FROM shipments s WHERE s.delivered_date IS NOT NULL AND s.shipped_date IS NOT NULL GROUP BY s.carrier ORDER BY avg_days ASC

-- predicted
SELECT s.carrier, AVG(DATEDIFF('day', o.order_date, s.delivered_date)) AS average_days FROM orders o JOIN shipments s ON s.order_id = o.order_id GROUP BY s.carrier ORDER BY average_days DESC
```

`execution_error` — function datediff(unknown, timestamp with time zone, timestamp with time zone) does not exist

**bench-001795** · `v15` · very_hard / logistics

> Which orders were shipped after their required date?

```sql
-- gold
SELECT o.order_id, o.required_date, s.shipped_date FROM orders o JOIN shipments s ON s.order_id = o.order_id WHERE s.shipped_date IS NOT NULL AND s.shipped_date::date > o.required_date ORDER BY o.order_id

-- predicted
SELECT order_id, customer_id, required_date, shipped_date FROM orders WHERE shipped_date IS NOT NULL AND shipped_date > required_date
```

`unknown_column` — column "shipped_date" does not exist

**bench-002909** · `v23` · very_hard / finance

> Calculate mean days from order to payment.

```sql
-- gold
SELECT AVG(EXTRACT(EPOCH FROM (p.payment_date - o.order_date)) / 86400.0) AS avg_days_to_payment FROM payments p JOIN orders o ON o.order_id = p.order_id WHERE p.status = 'completed'

-- predicted
SELECT AVG(DATEDIFF(payment.payment_date, order.order_date)) AS average_days FROM orders JOIN payments payment ON payment.order_id = orders.order_id
```

`syntax_error` — syntax error at or near "order"

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
| Schema mode | `full_schema_no_retrieval` (hash `d03619e711661bc5`) |
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
