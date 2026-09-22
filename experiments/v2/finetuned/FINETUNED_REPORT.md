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
| **Execution accuracy** | **68.43 %** |
| Correct | 310 |
| Executable SQL | 428 (94.48 %) |
| Ran but wrong answer | 118 (26.05 %) |
| Schema hallucination | 3.75 % |
| Mean generation latency | 4292 ms |
| Median generation latency | 4207 ms |
| SQL execution latency (median) | 28 ms |

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
| Ran, wrong answer | 26.05 % |
| Unknown column (hallucinated) | 1.32 % |
| Unknown table (hallucinated) | 2.43 % |
| Syntax error | 0.22 % |
| Other execution error | 1.10 % |
| Timeout | 0.00 % |
| No SQL produced | 0.00 % |
| Unparseable SQL | 0.44 % |
| Not read-only | 0.00 % |
| Generation failed | 0.00 % |

### Outcome distribution

Every example lands in exactly one bucket, so these sum to the total.

| Outcome | Count | Share | Meaning |
|---|---:|---:|---|
| `correct` | 310 | 68.4 % | Correct — result matched the gold answer. |
| `wrong_result` | 118 | 26.0 % | Ran successfully but returned a different result set than the gold query. The most dangerous failure: it looks like a valid answer. |
| `unknown_table` | 11 | 2.4 % | Referenced a table that does not exist — schema hallucination. |
| `unknown_column` | 6 | 1.3 % | Referenced a column that does not exist — schema hallucination. |
| `execution_error` | 5 | 1.1 % | Executed but raised another database error (type mismatch, ambiguous reference, grouping violation). |
| `invalid_sql` | 2 | 0.4 % | Could not be parsed as a single SQL statement. |
| `syntax_error` | 1 | 0.2 % | PostgreSQL rejected the SQL as malformed. |

---

## The headline number is an undercount — here is why

Many benchmark questions never say *which columns* to return. "Show orders
placed in 2023" is answered just as well by `SELECT *` as by the four columns
the gold query happens to name. Strict comparison marks those wrong even when
the rows are byte-identical.

| | Count | Share |
|---|---:|---:|
| Correct (exact match) | 310 | 68.43 % |
| Right rows, different columns | 0 | 0.00 % |
| Genuinely wrong rows | 118 | 26.05 % |
| **Projection-tolerant accuracy** | **310** | **68.43 %** |

Both numbers are real and both matter:

* **68.43 %** is the honest strict floor, and the
  figure Phase 10 must beat on identical terms.
* **68.43 %** is how often the model
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
| Examples with `ORDER BY` in gold | 248 |
| Correct with order respected | 125 |
| Strict accuracy | 50.40 % |

A gap between this and the headline number means the model selected the right
rows but sorted them wrongly.

---

## Results by difficulty

| Difficulty | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `easy` | 126 | 126 | 100.0 % | 100.0 % |
| `medium` | 63 | 61 | 96.8 % | 100.0 % |
| `hard` | 144 | 79 | 54.9 % | 92.4 % |
| `enterprise` | 84 | 39 | 46.4 % | 95.2 % |
| `very_hard` | 36 | 5 | 13.9 % | 72.2 % |

## Results by domain

| Domain | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `finance` | 66 | 59 | 89.4 % | 100.0 % |
| `logistics` | 162 | 118 | 72.8 % | 89.5 % |
| `catalogue` | 57 | 36 | 63.2 % | 86.0 % |
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
| `join` | 48 | 34 | 70.8 % | 100.0 % |
| `group_by` | 174 | 117 | 67.2 % | 93.7 % |
| `date_diff` | 6 | 4 | 66.7 % | 100.0 % |
| `business_rule` | 72 | 34 | 47.2 % | 94.4 % |
| `multi_join` | 99 | 44 | 44.4 % | 88.9 % |
| `order_by` | 78 | 31 | 39.7 % | 85.9 % |
| `sum` | 69 | 25 | 36.2 % | 84.1 % |
| `limit` | 66 | 22 | 33.3 % | 83.3 % |
| `subquery` | 6 | 2 | 33.3 % | 100.0 % |
| `churn` | 24 | 5 | 20.8 % | 100.0 % |
| `window` | 18 | 1 | 5.6 % | 55.6 % |
| `cross_domain` | 9 | 0 | 0.0 % | 66.7 % |
| `not_exists` | 6 | 0 | 0.0 % | 83.3 % |
| `partition` | 12 | 0 | 0.0 % | 33.3 % |
| `row_number` | 12 | 0 | 0.0 % | 33.3 % |

---

## Failure patterns

Classification of the 118 genuinely wrong answers, excluding the
column-projection artefact above.

| Pattern | Count | Share |
|---|---:|---:|
| Right row count, different values (usually a different metric definition) | 72 | 61 % |
| LEFT JOIN where gold used INNER (kept unmatched rows) | 21 | 18 % |
| Returned zero rows (over-filtered) | 13 | 11 % |
| Too many rows (under-filtered) | 4 | 3 % |
| Invented a LIMIT the question never asked for | 3 | 3 % |
| Too few rows (over-filtered) | 2 | 2 % |
| INNER JOIN where gold used LEFT (dropped rows) | 2 | 2 % |
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

**bench-001649** · `v02` · very_hard / catalogue

> For every category, list its 2 best selling products.

```sql
-- gold
WITH ranked AS ( SELECT p.category_id, p.product_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue, ROW_NUMBER() OVER ( PARTITION BY p.category_id ORDER BY SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) DESC ) AS rn FROM products p JOIN order_items oi ON oi.product_id = p.product_id GROUP BY p.category_id, p.product_id, p.product_name) SELECT category_

-- predicted
WITH product_sales AS ( SELECT p.product_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue FROM products p JOIN order_items oi ON oi.product_id = p.product_id GROUP BY p.product_id, p.product_name) SELECT c.category_name, p.product_name, ps.revenue FROM categories c JOIN products p ON p.category_id = c.category_id JOIN product_sales ps ON ps.product_id = p.product_id 
```

`unknown_column` — column ps.product_id does not exist

**bench-001650** · `v02` · very_hard / catalogue

> What are the 2 highest earning products per category?

```sql
-- gold
WITH ranked AS ( SELECT p.category_id, p.product_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue, ROW_NUMBER() OVER ( PARTITION BY p.category_id ORDER BY SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) DESC ) AS rn FROM products p JOIN order_items oi ON oi.product_id = p.product_id GROUP BY p.category_id, p.product_id, p.product_name) SELECT category_

-- predicted
WITH ranked AS ( SELECT c.category_name, p.product_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue FROM categories c JOIN products p ON p.category_id = c.category_id JOIN order_items oi ON oi.product_id = p.product_id GROUP BY c.category_id, c.category_name, p.product_id, p.product_name) SELECT category_name, product_name, revenue FROM ranked ORDER BY category_name,
```

`syntax_error` — syntax error at or near "GROUP"

**bench-001652** · `v02` · very_hard / catalogue

> For every category, list its 3 best selling products.

```sql
-- gold
WITH ranked AS ( SELECT p.category_id, p.product_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue, ROW_NUMBER() OVER ( PARTITION BY p.category_id ORDER BY SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) DESC ) AS rn FROM products p JOIN order_items oi ON oi.product_id = p.product_id GROUP BY p.category_id, p.product_id, p.product_name) SELECT category_

-- predicted
WITH ranked AS ( SELECT c.category_name, p.product_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue FROM categories c JOIN products p ON p.category_id = c.category_id JOIN order_items oi ON oi.product_id = p.product_id GROUP BY c.category_id, p.product_id, p.product_name, c.category_name ORDER BY c.category_id, revenue DESC) SELECT category_name, product_name, revenu
```

`execution_error` — WITHIN GROUP is required for ordered-set aggregate rank

**bench-002768** · `h32` · hard / logistics

> Show the top 2 warehouses by inventory value.

```sql
-- gold
SELECT w.warehouse_name, SUM(i.quantity_on_hand * p.unit_price) AS stock_value FROM inventory i JOIN warehouses w ON w.warehouse_id = i.warehouse_id JOIN products p ON p.product_id = i.product_id GROUP BY w.warehouse_id, w.warehouse_name ORDER BY stock_value DESC LIMIT 2

-- predicted
SELECT w.warehouse_name, SUM(pi.quantity_on_hand * p.unit_price) AS inventory_value FROM warehouses w JOIN inventory i ON i.warehouse_id = w.warehouse_id JOIN products p ON p.product_id = i.product_id GROUP BY w.warehouse_id, w.warehouse_name ORDER BY inventory_value DESC LIMIT 2
```

`unknown_table` — missing FROM-clause entry for table "pi"

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
