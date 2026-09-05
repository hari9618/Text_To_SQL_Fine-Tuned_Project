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
| **Execution accuracy** | **9.27 %** |
| Correct | 42 |
| Executable SQL | 418 (92.27 %) |
| Ran but wrong answer | 376 (83.00 %) |
| Schema hallucination | 3.31 % |
| Mean generation latency | 1951 ms |
| Median generation latency | 1738 ms |
| SQL execution latency (median) | 22 ms |

Latency is measured over first-attempt calls only (271
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
| Ran, wrong answer | 83.00 % |
| Unknown column (hallucinated) | 3.09 % |
| Unknown table (hallucinated) | 0.22 % |
| Syntax error | 0.00 % |
| Other execution error | 2.65 % |
| Timeout | 0.00 % |
| No SQL produced | 0.00 % |
| Unparseable SQL | 0.22 % |
| Not read-only | 0.00 % |
| Generation failed | 1.55 % |

### Outcome distribution

Every example lands in exactly one bucket, so these sum to the total.

| Outcome | Count | Share | Meaning |
|---|---:|---:|---|
| `wrong_result` | 376 | 83.0 % | Ran successfully but returned a different result set than the gold query. The most dangerous failure: it looks like a valid answer. |
| `correct` | 42 | 9.3 % | Correct — result matched the gold answer. |
| `unknown_column` | 14 | 3.1 % | Referenced a column that does not exist — schema hallucination. |
| `execution_error` | 12 | 2.6 % | Executed but raised another database error (type mismatch, ambiguous reference, grouping violation). |
| `generation_failed` | 7 | 1.5 % | The inference API call failed after all retries. |
| `unknown_table` | 1 | 0.2 % | Referenced a table that does not exist — schema hallucination. |
| `invalid_sql` | 1 | 0.2 % | Could not be parsed as a single SQL statement. |

---

## The headline number is an undercount — here is why

Many benchmark questions never say *which columns* to return. "Show orders
placed in 2023" is answered just as well by `SELECT *` as by the four columns
the gold query happens to name. Strict comparison marks those wrong even when
the rows are byte-identical.

| | Count | Share |
|---|---:|---:|
| Correct (exact match) | 42 | 9.27 % |
| Right rows, different columns | 157 | 34.66 % |
| Genuinely wrong rows | 219 | 48.34 % |
| **Projection-tolerant accuracy** | **199** | **43.93 %** |

Both numbers are real and both matter:

* **9.27 %** is the honest strict floor, and the
  figure Phase 10 must beat on identical terms.
* **43.93 %** is how often the model
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
| Examples with `ORDER BY` in gold | 243 |
| Correct with order respected | 8 |
| Strict accuracy | 3.29 % |

A gap between this and the headline number means the model selected the right
rows but sorted them wrongly.

---

## Results by difficulty

| Difficulty | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `medium` | 63 | 28 | 44.4 % | 90.5 % |
| `enterprise` | 84 | 11 | 13.1 % | 97.6 % |
| `hard` | 144 | 3 | 2.1 % | 86.1 % |
| `easy` | 126 | 0 | 0.0 % | 99.2 % |
| `very_hard` | 36 | 0 | 0.0 % | 83.3 % |

## Results by domain

| Domain | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `hr` | 9 | 3 | 33.3 % | 100.0 % |
| `finance` | 66 | 16 | 24.2 % | 95.5 % |
| `sales` | 156 | 16 | 10.3 % | 92.3 % |
| `catalogue` | 57 | 5 | 8.8 % | 96.5 % |
| `logistics` | 162 | 2 | 1.2 % | 88.9 % |
| `cross_domain` | 3 | 0 | 0.0 % | 100.0 % |

## Results by SQL pattern

Query features are not exclusive — one query can be counted under several.

| SQL feature | Examples | Correct | Execution accuracy | Executable |
|---|---:|---:|---:|---:|
| `count` | 57 | 21 | 36.8 % | 94.7 % |
| `null_handling` | 42 | 11 | 26.2 % | 100.0 % |
| `having` | 48 | 10 | 20.8 % | 95.8 % |
| `group_by` | 174 | 30 | 17.2 % | 86.2 % |
| `avg` | 48 | 8 | 16.7 % | 87.5 % |
| `not_exists` | 6 | 1 | 16.7 % | 100.0 % |
| `subquery` | 6 | 1 | 16.7 % | 83.3 % |
| `business_rule` | 72 | 9 | 12.5 % | 97.2 % |
| `order_by` | 78 | 7 | 9.0 % | 76.9 % |
| `join` | 48 | 2 | 4.2 % | 91.7 % |
| `date` | 111 | 4 | 3.6 % | 94.6 % |
| `sum` | 69 | 1 | 1.4 % | 72.5 % |
| `churn` | 24 | 0 | 0.0 % | 95.8 % |
| `cross_domain` | 9 | 0 | 0.0 % | 88.9 % |
| `date_diff` | 6 | 0 | 0.0 % | 50.0 % |
| `limit` | 66 | 0 | 0.0 % | 72.7 % |
| `multi_join` | 99 | 0 | 0.0 % | 80.8 % |
| `partition` | 12 | 0 | 0.0 % | 100.0 % |
| `row_number` | 12 | 0 | 0.0 % | 100.0 % |
| `select` | 126 | 0 | 0.0 % | 99.2 % |
| `where` | 162 | 0 | 0.0 % | 98.8 % |
| `window` | 18 | 0 | 0.0 % | 100.0 % |

---

## Failure patterns

Classification of the 219 genuinely wrong answers, excluding the
column-projection artefact above.

| Pattern | Count | Share |
|---|---:|---:|
| Right row count, different values (usually a different metric definition) | 129 | 59 % |
| LEFT JOIN where gold used INNER (kept unmatched rows) | 39 | 18 % |
| Invented a LIMIT the question never asked for | 19 | 9 % |
| Returned zero rows (over-filtered) | 13 | 6 % |
| Added DISTINCT, collapsing legitimate duplicates | 8 | 4 % |
| Too few rows (over-filtered) | 5 | 2 % |
| Too many rows (under-filtered) | 3 | 1 % |
| INNER JOIN where gold used LEFT (dropped rows) | 3 | 1 % |

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

**bench-000665** · `m08` · medium / sales

> Show yearly revenue totals.

```sql
-- gold
SELECT EXTRACT(YEAR FROM order_date) AS year, SUM(total_amount) AS revenue FROM orders GROUP BY year ORDER BY year

-- predicted
SELECT EXTRACT(YEAR FROM o.order_date) AS year, SUM((oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) + s.shipping_cost) AS yearly_revenue FROM orders o JOIN order_items oi ON o.order_id = oi.order_id JOIN shipments s ON o.order_id = s.order_id GROUP BY EXTRACT(YEAR FROM o.order_date) ORDER BY year
```

`unknown_column` — column s.shipping_cost does not exist

**bench-000884** · `h02` · hard / sales

> Show the 30 highest spending customers.

```sql
-- gold
SELECT c.customer_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue FROM customers c JOIN orders o ON o.customer_id = c.customer_id JOIN order_items oi ON oi.order_id = o.order_id GROUP BY c.customer_id, c.customer_name ORDER BY revenue DESC LIMIT 30

-- predicted
SELECT customer_id, customer_name, SUM(total_amount) AS total_spent FROM orders JOIN customers ON orders.customer_id = customers.customer_id GROUP BY customer_id, customer_name ORDER BY total_spent DESC LIMIT 30
```

`execution_error` — column reference "customer_id" is ambiguous

**bench-001530** · `h19` · hard / logistics

> List products held at Marseille.

```sql
-- gold
SELECT DISTINCT p.product_name, w.warehouse_name FROM products p JOIN inventory i ON i.product_id = p.product_id JOIN warehouses w ON w.warehouse_id = i.warehouse_id WHERE w.city = 'Marseille'

-- predicted
SELECT p.product_id, p.product_name, p.sku, p.unit_price, p.unit_cost, p.weight_kg, p.is_discontinued FROM products p JOIN warehouses w ON p.product_id = inventory.product_id JOIN inventory i ON i.warehouse_id = w.warehouse_id WHERE w.city = 'Marseille'
```

`unknown_table` — missing FROM-clause entry for table "inventory"

**bench-001853** · `x08` · enterprise / logistics

> Show stock that needs reordering.

```sql
-- gold
SELECT p.product_name, w.warehouse_name, i.quantity_on_hand, i.reorder_level FROM inventory i JOIN products p ON p.product_id = i.product_id JOIN warehouses w ON w.warehouse_id = i.warehouse_id WHERE i.quantity_on_hand < i.reorder_level ORDER BY p.product_name

-- predicted
(nothing produced)
```

`generation_failed` — HfHubHTTPError: 402 Client Error: Payment Required for url: https://router.huggingface.co/nscale/v1/chat/completions (Request ID: Root=1-6a8db0c5-25a0ec511fb7d08f698c6f7e;1d7f39cd-e1d8-4a01-9fa8-7d1c0

**bench-002244** · `x14` · enterprise / sales

> List churned customers with no order in 17 months.

```sql
-- gold
SELECT c.customer_id, c.customer_name, MAX(o.order_date) AS last_order FROM customers c JOIN orders o ON o.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name HAVING MAX(o.order_date) < TIMESTAMPTZ '2026-08-01 00:00:00+00:00' - INTERVAL '17 months' ORDER BY last_order

-- predicted
SELECT DISTINCT c.customer_id, c.customer_name FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id WHERE c.is_active = false AND o.order_date IS NULL OR (o.order_date IS NOT NULL AND o.order_date < CURRENT_DATE - INTERVAL '17 months'
```

`invalid_sql` — Expecting ). Line 1, Col: 250.

---

## Reproducibility

| | |
|---|---|
| Model | `Qwen/Qwen3-8B` |
| Model revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Serving | hf_inference_providers / featherless-ai |
| Fine-tuned | False |
| Adapters | none |
| Inference params | `{'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 512, 'seed': 20260808}` |
| Prompt version | `v1` (hash `8288e41a496531a9`) |
| Schema mode | `retrieved_keyword_v1` (hash `n/a - retrieved per question`) |
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
