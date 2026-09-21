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
| **Execution accuracy** | **43.71 %** |
| Correct | 198 |
| Executable SQL | 433 (95.58 %) |
| Ran but wrong answer | 235 (51.88 %) |
| Schema hallucination | 0.22 % |
| Mean generation latency | 4173 ms |
| Median generation latency | 3997 ms |
| SQL execution latency (median) | 11 ms |

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
| Ran, wrong answer | 51.88 % |
| Unknown column (hallucinated) | 0.22 % |
| Unknown table (hallucinated) | 0.00 % |
| Syntax error | 0.00 % |
| Other execution error | 4.19 % |
| Timeout | 0.00 % |
| No SQL produced | 0.00 % |
| Unparseable SQL | 0.00 % |
| Not read-only | 0.00 % |
| Generation failed | 0.00 % |

### Outcome distribution

Every example lands in exactly one bucket, so these sum to the total.

| Outcome | Count | Share | Meaning |
|---|---:|---:|---|
| `wrong_result` | 235 | 51.9 % | Ran successfully but returned a different result set than the gold query. The most dangerous failure: it looks like a valid answer. |
| `correct` | 198 | 43.7 % | Correct — result matched the gold answer. |
| `execution_error` | 19 | 4.2 % | Executed but raised another database error (type mismatch, ambiguous reference, grouping violation). |
| `unknown_column` | 1 | 0.2 % | Referenced a column that does not exist — schema hallucination. |

---

## The headline number is an undercount — here is why

Many benchmark questions never say *which columns* to return. "Show orders
placed in 2023" is answered just as well by `SELECT *` as by the four columns
the gold query happens to name. Strict comparison marks those wrong even when
the rows are byte-identical.

| | Count | Share |
|---|---:|---:|
| Correct (exact match) | 198 | 43.71 % |
| Right rows, different columns | 18 | 3.97 % |
| Genuinely wrong rows | 217 | 47.90 % |
| **Projection-tolerant accuracy** | **216** | **47.68 %** |

Both numbers are real and both matter:

* **43.71 %** is the honest strict floor, and the
  figure Phase 10 must beat on identical terms.
* **47.68 %** is how often the model
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
| Examples with `ORDER BY` in gold | 255 |
| Correct with order respected | 19 |
| Strict accuracy | 7.45 % |

A gap between this and the headline number means the model selected the right
rows but sorted them wrongly.

---

## Results by difficulty

| Difficulty | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `easy` | 126 | 117 | 92.9 % | 100.0 % |
| `medium` | 63 | 44 | 69.8 % | 100.0 % |
| `enterprise` | 84 | 26 | 30.9 % | 100.0 % |
| `hard` | 144 | 11 | 7.6 % | 88.2 % |
| `very_hard` | 36 | 0 | 0.0 % | 91.7 % |

## Results by domain

| Domain | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `finance` | 66 | 57 | 86.4 % | 97.0 % |
| `logistics` | 162 | 64 | 39.5 % | 99.4 % |
| `sales` | 156 | 58 | 37.2 % | 89.1 % |
| `hr` | 9 | 3 | 33.3 % | 100.0 % |
| `catalogue` | 57 | 16 | 28.1 % | 100.0 % |
| `cross_domain` | 3 | 0 | 0.0 % | 100.0 % |

## Results by SQL pattern

Query features are not exclusive — one query can be counted under several.

| SQL feature | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `select` | 126 | 117 | 92.9 % | 100.0 % |
| `where` | 162 | 117 | 72.2 % | 100.0 % |
| `count` | 57 | 36 | 63.2 % | 100.0 % |
| `null_handling` | 42 | 26 | 61.9 % | 100.0 % |
| `date` | 111 | 56 | 50.5 % | 100.0 % |
| `having` | 48 | 24 | 50.0 % | 100.0 % |
| `avg` | 48 | 19 | 39.6 % | 89.6 % |
| `group_by` | 174 | 55 | 31.6 % | 92.0 % |
| `business_rule` | 72 | 18 | 25.0 % | 100.0 % |
| `join` | 48 | 11 | 22.9 % | 95.8 % |
| `order_by` | 78 | 7 | 9.0 % | 82.0 % |
| `churn` | 24 | 0 | 0.0 % | 100.0 % |
| `cross_domain` | 9 | 0 | 0.0 % | 100.0 % |
| `date_diff` | 6 | 0 | 0.0 % | 66.7 % |
| `limit` | 66 | 0 | 0.0 % | 78.8 % |
| `multi_join` | 99 | 0 | 0.0 % | 85.9 % |
| `not_exists` | 6 | 0 | 0.0 % | 100.0 % |
| `partition` | 12 | 0 | 0.0 % | 100.0 % |
| `row_number` | 12 | 0 | 0.0 % | 100.0 % |
| `subquery` | 6 | 0 | 0.0 % | 50.0 % |
| `sum` | 69 | 0 | 0.0 % | 79.7 % |
| `window` | 18 | 0 | 0.0 % | 100.0 % |

---

## Failure patterns

Classification of the 217 genuinely wrong answers, excluding the
column-projection artefact above.

| Pattern | Count | Share |
|---|---:|---:|
| Right row count, different values (usually a different metric definition) | 121 | 56 % |
| Returned zero rows (over-filtered) | 35 | 16 % |
| Added DISTINCT, collapsing legitimate duplicates | 26 | 12 % |
| LEFT JOIN where gold used INNER (kept unmatched rows) | 14 | 6 % |
| Too few rows (over-filtered) | 11 | 5 % |
| Too many rows (under-filtered) | 6 | 3 % |
| Invented a LIMIT the question never asked for | 2 | 1 % |
| INNER JOIN where gold used LEFT (dropped rows) | 2 | 1 % |

The largest bucket is *not* broken SQL. It is the model computing a defensible
but different quantity — most often revenue summed from `order_items` lines
when the gold query used the denormalised `orders.total_amount`. Both are
reasonable readings of "revenue"; the schema supports two answers and the
question does not disambiguate. That is a business-rule gap, which is exactly
what fine-tuning on this domain is expected to close.

---

## Successful examples

**bench-000322** · `e13` · easy / sales

> Show orders placed in 2023.

```sql
-- gold
SELECT * FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023

-- predicted
SELECT * FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023
```

**bench-000323** · `e13` · easy / sales

> List all orders from the year 2023.

```sql
-- gold
SELECT * FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023

-- predicted
SELECT * FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023
```

**bench-000324** · `e13` · easy / sales

> Which orders were made during 2023?

```sql
-- gold
SELECT * FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023

-- predicted
SELECT * FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023
```

**bench-000325** · `e13` · easy / sales

> Show orders placed in 2026.

```sql
-- gold
SELECT * FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2026

-- predicted
SELECT * FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2026
```

---

## Failed examples

**bench-000565** · `e22` · easy / logistics

> Show shipments handled by GlobalEx.

```sql
-- gold
SELECT * FROM shipments WHERE carrier = 'GlobalEx'

-- predicted
SELECT * FROM shipments JOIN orders ON shipments.order_id = orders.order_id JOIN customers ON orders.customer_id = customers.customer_id WHERE customers.customer_name = 'GlobalEx'
```

`wrong_result` — gold returned 8253 rows, prediction returned 0

**bench-000878** · `h02` · hard / sales

> Show the 15 highest spending customers.

```sql
-- gold
SELECT c.customer_name, SUM(o.total_amount) AS revenue FROM customers c JOIN orders o ON o.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name ORDER BY revenue DESC LIMIT 15

-- predicted
SELECT customer_id, customer_name, SUM(total_amount) AS total_spent FROM orders JOIN customers ON orders.customer_id = customers.customer_id WHERE status = 'completed' GROUP BY customer_id, customer_name ORDER BY total_spent DESC LIMIT 15
```

`execution_error` — column reference "customer_id" is ambiguous

**bench-001795** · `v15` · very_hard / logistics

> Which orders were shipped after their required date?

```sql
-- gold
SELECT o.order_id, o.required_date, s.shipped_date FROM orders o JOIN shipments s ON s.order_id = o.order_id WHERE s.shipped_date IS NOT NULL AND s.shipped_date::date > o.required_date ORDER BY o.order_id

-- predicted
SELECT * FROM orders WHERE shipped_date IS NOT NULL AND shipped_date > required_date
```

`unknown_column` — column "shipped_date" does not exist

**bench-000571** · `e22` · easy / logistics

> Show shipments handled by BluePort.

```sql
-- gold
SELECT * FROM shipments WHERE carrier = 'BluePort'

-- predicted
SELECT * FROM shipments JOIN orders ON shipments.order_id = orders.order_id JOIN customers ON orders.customer_id = customers.customer_id WHERE customers.customer_name = 'BluePort'
```

`wrong_result` — gold returned 8216 rows, prediction returned 0

**bench-000664** · `m08` · medium / sales

> What is the total revenue for each year?

```sql
-- gold
SELECT EXTRACT(YEAR FROM order_date) AS year, SUM(total_amount) AS revenue FROM orders GROUP BY year ORDER BY year

-- predicted
SELECT EXTRACT(YEAR FROM orders.order_date) AS year, SUM(order_items.quantity * order_items.unit_price * (1 - order_items.discount_pct / 100)) AS total_revenue FROM orders JOIN order_items ON orders.order_id = order_items.order_id WHERE orders.status = 'completed' AND orders.total_amount = (SELECT SUM(quantity * unit_price * (1 - discount_pct / 100)) FROM order_items WHERE order_id = orders.order_
```

`wrong_result` — gold returned 8 rows, prediction returned 0

**bench-000665** · `m08` · medium / sales

> Show yearly revenue totals.

```sql
-- gold
SELECT EXTRACT(YEAR FROM order_date) AS year, SUM(total_amount) AS revenue FROM orders GROUP BY year ORDER BY year

-- predicted
SELECT EXTRACT(YEAR FROM orders.order_date) AS year, SUM(order_items.quantity * order_items.unit_price * (1 - order_items.discount_pct / 100)) AS yearly_revenue FROM orders JOIN order_items ON orders.order_id = order_items.order_id WHERE orders.status = 'completed' GROUP BY EXTRACT(YEAR FROM orders.order_date)
```

`wrong_result` — gold returned 8 rows, prediction returned 0

---

## Reproducibility

| | |
|---|---|
| Model | `Qwen/Qwen3-8B` |
| Model revision | `b968826d9c46` |
| Serving | base_model_4bit_replayed / kaggle-t4-local-generation |
| Fine-tuned | False |
| Adapters | none |
| Inference params | `None` |
| Prompt version | `v2` (hash `4e72cc5f722ce436`) |
| Schema mode | `full_schema_no_retrieval` (hash `d03619e711661bc5`) |
| Dataset | `C:/Users/dell/Desktop/Enterprice-text to -SQL/dataset/v2/test/test.jsonl` (hash `d7ec17963f69c87d`) |
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
