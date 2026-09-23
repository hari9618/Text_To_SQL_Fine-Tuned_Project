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
| **Execution accuracy** | **70.86 %** |
| Correct | 321 |
| Executable SQL | 445 (98.23 %) |
| Ran but wrong answer | 124 (27.37 %) |
| Schema hallucination | 0.66 % |
| Mean generation latency | 5145 ms |
| Median generation latency | 4207 ms |
| SQL execution latency (median) | 19 ms |

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
| Ran, wrong answer | 27.37 % |
| Unknown column (hallucinated) | 0.66 % |
| Unknown table (hallucinated) | 0.00 % |
| Syntax error | 0.22 % |
| Other execution error | 0.22 % |
| Timeout | 0.00 % |
| No SQL produced | 0.00 % |
| Unparseable SQL | 0.66 % |
| Not read-only | 0.00 % |
| Generation failed | 0.00 % |

### Outcome distribution

Every example lands in exactly one bucket, so these sum to the total.

| Outcome | Count | Share | Meaning |
|---|---:|---:|---|
| `correct` | 321 | 70.9 % | Correct — result matched the gold answer. |
| `wrong_result` | 124 | 27.4 % | Ran successfully but returned a different result set than the gold query. The most dangerous failure: it looks like a valid answer. |
| `invalid_sql` | 3 | 0.7 % | Could not be parsed as a single SQL statement. |
| `unknown_column` | 3 | 0.7 % | Referenced a column that does not exist — schema hallucination. |
| `syntax_error` | 1 | 0.2 % | PostgreSQL rejected the SQL as malformed. |
| `execution_error` | 1 | 0.2 % | Executed but raised another database error (type mismatch, ambiguous reference, grouping violation). |

---

## The headline number is an undercount — here is why

Many benchmark questions never say *which columns* to return. "Show orders
placed in 2023" is answered just as well by `SELECT *` as by the four columns
the gold query happens to name. Strict comparison marks those wrong even when
the rows are byte-identical.

| | Count | Share |
|---|---:|---:|
| Correct (exact match) | 321 | 70.86 % |
| Right rows, different columns | 0 | 0.00 % |
| Genuinely wrong rows | 124 | 27.37 % |
| **Projection-tolerant accuracy** | **321** | **70.86 %** |

Both numbers are real and both matter:

* **70.86 %** is the honest strict floor, and the
  figure Phase 10 must beat on identical terms.
* **70.86 %** is how often the model
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
| Examples with `ORDER BY` in gold | 265 |
| Correct with order respected | 136 |
| Strict accuracy | 51.32 % |

A gap between this and the headline number means the model selected the right
rows but sorted them wrongly.

---

## Results by difficulty

| Difficulty | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `easy` | 126 | 126 | 100.0 % | 100.0 % |
| `medium` | 63 | 61 | 96.8 % | 100.0 % |
| `hard` | 144 | 90 | 62.5 % | 100.0 % |
| `enterprise` | 84 | 39 | 46.4 % | 97.6 % |
| `very_hard` | 36 | 5 | 13.9 % | 83.3 % |

## Results by domain

| Domain | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `finance` | 66 | 59 | 89.4 % | 100.0 % |
| `logistics` | 162 | 129 | 79.6 % | 97.5 % |
| `catalogue` | 57 | 36 | 63.2 % | 93.0 % |
| `sales` | 156 | 93 | 59.6 % | 100.0 % |
| `hr` | 9 | 4 | 44.4 % | 100.0 % |
| `cross_domain` | 3 | 0 | 0.0 % | 100.0 % |

## Results by SQL pattern

Query features are not exclusive — one query can be counted under several.

| SQL feature | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `select` | 126 | 126 | 100.0 % | 100.0 % |
| `count` | 57 | 53 | 93.0 % | 100.0 % |
| `where` | 162 | 150 | 92.6 % | 100.0 % |
| `avg` | 48 | 43 | 89.6 % | 100.0 % |
| `having` | 48 | 42 | 87.5 % | 100.0 % |
| `null_handling` | 42 | 32 | 76.2 % | 100.0 % |
| `date` | 111 | 84 | 75.7 % | 100.0 % |
| `group_by` | 174 | 128 | 73.6 % | 100.0 % |
| `join` | 48 | 34 | 70.8 % | 100.0 % |
| `date_diff` | 6 | 4 | 66.7 % | 100.0 % |
| `multi_join` | 99 | 55 | 55.6 % | 100.0 % |
| `order_by` | 78 | 42 | 53.9 % | 100.0 % |
| `sum` | 69 | 36 | 52.2 % | 100.0 % |
| `limit` | 66 | 33 | 50.0 % | 100.0 % |
| `business_rule` | 72 | 34 | 47.2 % | 97.2 % |
| `subquery` | 6 | 2 | 33.3 % | 100.0 % |
| `churn` | 24 | 5 | 20.8 % | 100.0 % |
| `window` | 18 | 1 | 5.6 % | 77.8 % |
| `cross_domain` | 9 | 0 | 0.0 % | 88.9 % |
| `not_exists` | 6 | 0 | 0.0 % | 83.3 % |
| `partition` | 12 | 0 | 0.0 % | 66.7 % |
| `row_number` | 12 | 0 | 0.0 % | 66.7 % |

---

## Failure patterns

Classification of the 124 genuinely wrong answers, excluding the
column-projection artefact above.

| Pattern | Count | Share |
|---|---:|---:|
| Right row count, different values (usually a different metric definition) | 75 | 60 % |
| LEFT JOIN where gold used INNER (kept unmatched rows) | 21 | 17 % |
| Returned zero rows (over-filtered) | 13 | 10 % |
| Too many rows (under-filtered) | 5 | 4 % |
| INNER JOIN where gold used LEFT (dropped rows) | 4 | 3 % |
| Invented a LIMIT the question never asked for | 3 | 2 % |
| Too few rows (over-filtered) | 2 | 2 % |
| Added DISTINCT, collapsing legitimate duplicates | 1 | 1 % |

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

**bench-000790** · `m14` · medium / catalogue

> What is the average product price in each category?

```sql
-- gold
SELECT category_id, AVG(unit_price) AS average_price FROM products GROUP BY category_id ORDER BY average_price DESC

-- predicted
SELECT c.category_name, AVG(p.unit_price) AS average_price FROM categories c JOIN products p ON p.category_id = c.category_id GROUP BY c.category_id, c.category_name ORDER BY average_price DESC
```

`wrong_result` — gold returned 110 rows, prediction returned 110

**bench-001648** · `v02` · very_hard / catalogue

> Show the top 2 products by revenue within each category.

```sql
-- gold
WITH ranked AS ( SELECT p.category_id, p.product_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue, ROW_NUMBER() OVER ( PARTITION BY p.category_id ORDER BY SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) DESC ) AS rn FROM products p JOIN order_items oi ON oi.product_id = p.product_id GROUP BY p.category_id, p.product_id, p.product_name) SELECT category_

-- predicted
WITH ranked AS ( SELECT c.category_name, p.product_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue FROM categories c JOIN products p ON p.category_id = c.category_id JOIN order_items oi ON oi.product_id = p.product_id GROUP BY c.category_id, c.category_name, p.product_id, p.product_name) SELECT category_name, product_name, revenue FROM ranked ORDER BY category_name,
```

`invalid_sql` — Invalid expression / Unexpected token. Line 1, Col: 414.

**bench-001659** · `v02` · very_hard / catalogue

> What are the 4 highest earning products per category?

```sql
-- gold
WITH ranked AS ( SELECT p.category_id, p.product_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue, ROW_NUMBER() OVER ( PARTITION BY p.category_id ORDER BY SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) DESC ) AS rn FROM products p JOIN order_items oi ON oi.product_id = p.product_id GROUP BY p.category_id, p.product_id, p.product_name) SELECT category_

-- predicted
WITH ranked AS ( SELECT c.category_name, p.product_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue FROM categories c JOIN products p ON p.category_id = c.category_id JOIN order_items oi ON oi.product_id = p.product_id GROUP BY c.category_id, p.product_id, p.product_name, c.category_name ORDER BY c.category_name, revenue DESC) SELECT category_name, product_name, reve
```

`syntax_error` — syntax error at or near "WHERE"

**bench-001795** · `v15` · very_hard / logistics

> Which orders were shipped after their required date?

```sql
-- gold
SELECT o.order_id, o.required_date, s.shipped_date FROM orders o JOIN shipments s ON s.order_id = o.order_id WHERE s.shipped_date IS NOT NULL AND s.shipped_date::date > o.required_date ORDER BY o.order_id

-- predicted
SELECT * FROM orders WHERE shipped_date IS NOT NULL AND shipped_date > required_date ORDER BY shipped_date
```

`unknown_column` — column "shipped_date" does not exist

**bench-002251** · `x15` · enterprise / logistics

> How full is each warehouse relative to its capacity?

```sql
-- gold
SELECT w.warehouse_name, w.capacity_units, COALESCE(SUM(i.quantity_on_hand), 0) AS stock_held, ROUND(100.0 * COALESCE(SUM(i.quantity_on_hand), 0) / NULLIF(w.capacity_units, 0), 2) AS utilisation_pct FROM warehouses w LEFT JOIN inventory i ON i.warehouse_id = w.warehouse_id GROUP BY w.warehouse_id, w.warehouse_name, w.capacity_units ORDER BY utilisation_pct DESC

-- predicted
SELECT warehouse_id, SUM(quantity_on_hand) AS total_stock, SUM(quantity_on_hand) * 1.0 / SUM(capacity_units) AS utilization FROM inventory JOIN warehouses ON warehouses.warehouse_id = inventory.warehouse_id GROUP BY warehouse_id ORDER BY utilization DESC
```

`execution_error` — column reference "warehouse_id" is ambiguous

**bench-000792** · `m14` · medium / catalogue

> Calculate average price per category.

```sql
-- gold
SELECT category_id, AVG(unit_price) AS average_price FROM products GROUP BY category_id ORDER BY average_price DESC

-- predicted
SELECT c.category_name, AVG(p.unit_price) AS average_price FROM categories c JOIN products p ON p.category_id = c.category_id GROUP BY c.category_id, c.category_name ORDER BY average_price DESC
```

`wrong_result` — gold returned 110 rows, prediction returned 110

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
