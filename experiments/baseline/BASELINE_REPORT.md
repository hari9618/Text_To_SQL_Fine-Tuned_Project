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
| **Execution accuracy** | **10.82 %** |
| Correct | 49 |
| Executable SQL | 448 (98.90 %) |
| Ran but wrong answer | 399 (88.08 %) |
| Schema hallucination | 0.66 % |
| Mean generation latency | 1330 ms |
| Median generation latency | 1233 ms |
| SQL execution latency (median) | 13 ms |

Latency is measured over first-attempt calls only (97
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
| Ran, wrong answer | 88.08 % |
| Unknown column (hallucinated) | 0.66 % |
| Unknown table (hallucinated) | 0.00 % |
| Syntax error | 0.00 % |
| Other execution error | 0.22 % |
| Timeout | 0.00 % |
| No SQL produced | 0.00 % |
| Unparseable SQL | 0.22 % |
| Not read-only | 0.00 % |
| Generation failed | 0.00 % |

### Outcome distribution

Every example lands in exactly one bucket, so these sum to the total.

| Outcome | Count | Share | Meaning |
|---|---:|---:|---|
| `wrong_result` | 399 | 88.1 % | Ran successfully but returned a different result set than the gold query. The most dangerous failure: it looks like a valid answer. |
| `correct` | 49 | 10.8 % | Correct — result matched the gold answer. |
| `unknown_column` | 3 | 0.7 % | Referenced a column that does not exist — schema hallucination. |
| `invalid_sql` | 1 | 0.2 % | Could not be parsed as a single SQL statement. |
| `execution_error` | 1 | 0.2 % | Executed but raised another database error (type mismatch, ambiguous reference, grouping violation). |

---

## The headline number is an undercount — here is why

Many benchmark questions never say *which columns* to return. "Show orders
placed in 2023" is answered just as well by `SELECT *` as by the four columns
the gold query happens to name. Strict comparison marks those wrong even when
the rows are byte-identical.

| | Count | Share |
|---|---:|---:|
| Correct (exact match) | 49 | 10.82 % |
| Right rows, different columns | 159 | 35.10 % |
| Genuinely wrong rows | 240 | 52.98 % |
| **Projection-tolerant accuracy** | **208** | **45.92 %** |

Both numbers are real and both matter:

* **10.82 %** is the honest strict floor, and the
  figure Phase 10 must beat on identical terms.
* **45.92 %** is how often the model
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
| Examples with `ORDER BY` in gold | 269 |
| Correct with order respected | 14 |
| Strict accuracy | 5.20 % |

A gap between this and the headline number means the model selected the right
rows but sorted them wrongly.

---

## Results by difficulty

| Difficulty | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `medium` | 63 | 33 | 52.4 % | 100.0 % |
| `enterprise` | 84 | 11 | 13.1 % | 98.8 % |
| `hard` | 144 | 5 | 3.5 % | 100.0 % |
| `easy` | 126 | 0 | 0.0 % | 100.0 % |
| `very_hard` | 36 | 0 | 0.0 % | 88.9 % |

## Results by domain

| Domain | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `hr` | 9 | 3 | 33.3 % | 100.0 % |
| `finance` | 66 | 11 | 16.7 % | 98.5 % |
| `sales` | 156 | 22 | 14.1 % | 100.0 % |
| `catalogue` | 57 | 7 | 12.3 % | 98.2 % |
| `logistics` | 162 | 6 | 3.7 % | 98.8 % |
| `cross_domain` | 3 | 0 | 0.0 % | 66.7 % |

## Results by SQL pattern

Query features are not exclusive — one query can be counted under several.

| SQL feature | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `count` | 57 | 25 | 43.9 % | 100.0 % |
| `not_exists` | 6 | 2 | 33.3 % | 100.0 % |
| `subquery` | 6 | 2 | 33.3 % | 100.0 % |
| `null_handling` | 42 | 10 | 23.8 % | 100.0 % |
| `avg` | 48 | 10 | 20.8 % | 97.9 % |
| `group_by` | 174 | 36 | 20.7 % | 100.0 % |
| `business_rule` | 72 | 10 | 13.9 % | 98.6 % |
| `date` | 111 | 12 | 10.8 % | 100.0 % |
| `having` | 48 | 5 | 10.4 % | 100.0 % |
| `order_by` | 78 | 7 | 9.0 % | 100.0 % |
| `join` | 48 | 3 | 6.2 % | 95.8 % |
| `sum` | 69 | 1 | 1.4 % | 100.0 % |
| `where` | 162 | 1 | 0.6 % | 100.0 % |
| `churn` | 24 | 0 | 0.0 % | 100.0 % |
| `cross_domain` | 9 | 0 | 0.0 % | 88.9 % |
| `date_diff` | 6 | 0 | 0.0 % | 83.3 % |
| `limit` | 66 | 0 | 0.0 % | 100.0 % |
| `multi_join` | 99 | 0 | 0.0 % | 100.0 % |
| `partition` | 12 | 0 | 0.0 % | 91.7 % |
| `row_number` | 12 | 0 | 0.0 % | 91.7 % |
| `select` | 126 | 0 | 0.0 % | 100.0 % |
| `window` | 18 | 0 | 0.0 % | 94.4 % |

---

## Failure patterns

Classification of the 240 genuinely wrong answers, excluding the
column-projection artefact above.

| Pattern | Count | Share |
|---|---:|---:|
| Right row count, different values (usually a different metric definition) | 149 | 62 % |
| LEFT JOIN where gold used INNER (kept unmatched rows) | 31 | 13 % |
| Returned zero rows (over-filtered) | 29 | 12 % |
| Invented a LIMIT the question never asked for | 19 | 8 % |
| Too many rows (under-filtered) | 5 | 2 % |
| Added DISTINCT, collapsing legitimate duplicates | 4 | 2 % |
| Too few rows (over-filtered) | 3 | 1 % |

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
SELECT status, COUNT(*) AS order_count FROM orders GROUP BY status
```

**bench-000653** · `m04` · medium / sales

> Count orders by status.

```sql
-- gold
SELECT status, COUNT(*) AS order_count FROM orders GROUP BY status ORDER BY order_count DESC

-- predicted
SELECT status, COUNT(*) AS order_count FROM orders GROUP BY status
```

**bench-000654** · `m04` · medium / sales

> Show the number of orders per status.

```sql
-- gold
SELECT status, COUNT(*) AS order_count FROM orders GROUP BY status ORDER BY order_count DESC

-- predicted
SELECT status, COUNT(*) AS order_count FROM orders GROUP BY status
```

**bench-000666** · `m08` · medium / sales

> Break down total order value by year.

```sql
-- gold
SELECT EXTRACT(YEAR FROM order_date) AS year, SUM(total_amount) AS revenue FROM orders GROUP BY year ORDER BY year

-- predicted
SELECT EXTRACT(YEAR FROM o.order_date) AS year, SUM(o.total_amount) AS total_order_value FROM orders o GROUP BY EXTRACT(YEAR FROM o.order_date) ORDER BY year
```

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

**bench-001655** · `v02` · very_hard / catalogue

> For every category, list its 5 best selling products.

```sql
-- gold
WITH ranked AS ( SELECT p.category_id, p.product_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue, ROW_NUMBER() OVER ( PARTITION BY p.category_id ORDER BY SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) DESC ) AS rn FROM products p JOIN order_items oi ON oi.product_id = p.product_id GROUP BY p.category_id, p.product_id, p.product_name) SELECT category_

-- predicted
SELECT c.category_id, c.category_name, p.product_id, p.product_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS total_revenue FROM categories c JOIN products p ON c.category_id = p.category_id JOIN order_items oi ON p.product_id = oi.product_id GROUP BY c.category_id, c.category_name, p.product_id, p.product_name ORDER BY c.category_id, total_revenue DESC LIMIT 1 PER GROUP
```

`invalid_sql` — Invalid expression / Unexpected token. Line 1, Col: 392.

**bench-001795** · `v15` · very_hard / logistics

> Which orders were shipped after their required date?

```sql
-- gold
SELECT o.order_id, o.required_date, s.shipped_date FROM orders o JOIN shipments s ON s.order_id = o.order_id WHERE s.shipped_date IS NOT NULL AND s.shipped_date::date > o.required_date ORDER BY o.order_id

-- predicted
SELECT order_id FROM orders WHERE shipped_date > required_date
```

`unknown_column` — column "shipped_date" does not exist

**bench-002909** · `v23` · very_hard / finance

> Calculate mean days from order to payment.

```sql
-- gold
SELECT AVG(EXTRACT(EPOCH FROM (p.payment_date - o.order_date)) / 86400.0) AS avg_days_to_payment FROM payments p JOIN orders o ON o.order_id = p.order_id WHERE p.status = 'completed'

-- predicted
SELECT AVG(DATEDIFF('day', o.order_date, p.payment_date)) AS mean_days_from_order_to_payment FROM orders o JOIN payments p ON o.order_id = p.order_id
```

`execution_error` — function datediff(unknown, timestamp with time zone, timestamp with time zone) does not exist

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

### Gold fingerprint check

All gold queries were re-executed and reproduced their Phase 3 fingerprints exactly — the database has not drifted since validation.

---

## What this number is not

This is the floor, not a verdict on the model. It has no schema retrieval, no
SQL repair, no few-shot examples, and no fine-tuning. Phases 6, 9 and 11 each
add one of those, and each is measured against this figure.

No claim that fine-tuning helps can be made until Phase 10 produces a
comparable number from an identical harness.
