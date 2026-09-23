# Failure analysis

Configuration: **fine-tuned + repair (configuration 5)** — `experiments/v2/repair/results.jsonl`

453 test questions, **321 correct (70.9 %)**, 132 failures. Every failure was re-executed against PostgreSQL next to its gold query and sorted into one bucket.

> **This is a post-hoc diagnostic on the test set.** It explains the frozen
> number; it does not change it. Any fix it motivates must be chosen on the
> validation split and re-measured as a new configuration.

## Where the failures are

| bucket | n | of failures | of test set | meaning |
| --- | ---: | ---: | ---: | ---: |
| `projection_only` | 33 | 25.0 % | 7.3 % | right rows, different column set — the question never said which columns |
| `wrong_values` | 7 | 5.3 % | 1.5 % | same rows identified, different values — a bad computation |
| `wrong_rows` | 84 | 63.6 % | 18.5 % | genuinely different rows — a real logic error |
| `schema_hallucination` | 3 | 2.3 % | 0.7 % | referenced a table or column that does not exist |
| `execution_error` | 4 | 3.0 % | 0.9 % | ran but PostgreSQL rejected it (type error, ambiguity, ...) |
| `syntax_error` | 1 | 0.8 % | 0.2 % | did not parse |

Read it as three groups:

* **33 benign** (`projection_only` + `column_order`) — the rows are right. The question did not say which columns to return and the model chose a different set from the gold query. 25.0 % of all failures.
* **91 real logic errors** (`wrong_rows` + `wrong_values`) — the model misunderstood the question or the schema. 68.9 % of failures.
* **8 detectable** (hallucination, execution, syntax) — the only failures repair can see. 6.1 % of failures.

### Row-level accuracy (diagnostic, not the headline)

If a question does not specify columns and the model returns the right rows under a different column set, that is arguably correct. Counting `projection_only` and `column_order` as correct gives:

|  | strict (headline) | row-level (diagnostic) |
| --- | ---: | ---: |
| all | 70.9 % | 78.1 % |
| easy | 100.0 % | 100.0 % |
| medium | 96.8 % | 100.0 % |
| hard | 62.5 % | 71.5 % |
| very_hard | 13.9 % | 41.7 % |
| enterprise | 46.4 % | 56.0 % |

The strict number stays the headline: the benchmark was frozen before any of this was looked at, and loosening a metric after seeing test results is exactly the move this project refuses to make. The row-level figure is here so the gap between the two is visible and explained, not hidden.

## By difficulty

| tier | n | correct | `projection_only` | `wrong_values` | `wrong_rows` | `schema_hallucination` | `execution_error` | `syntax_error` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 126 | 126 |  |  |  |  |  |  |
| medium | 63 | 61 | 2 |  |  |  |  |  |
| hard | 144 | 90 | 13 | 6 | 35 |  |  |  |
| very_hard | 36 | 5 | 10 | 1 | 14 | 2 | 3 | 1 |
| enterprise | 84 | 39 | 8 |  | 35 | 1 | 1 |  |

## `projection_only`: which templates

Every question generated from the same template fails the same way, which is the signature of a convention mismatch rather than a model error — the model cannot know which four columns an unseen template's author chose.

| template | failed / total | example question | difference |
| --- | ---: | ---: | ---: |
| `h19` | 9 / 33 | List products held at Marseille. | `swapped warehouse_name -> quantity_on_hand` |
| `v02` | 3 / 12 | For every category, list its 3 best selling products. | `swapped category_id -> category_name` |
| `v16` | 3 / 3 | Show each customer's first order date and their total order count. | `swapped customer_name -> customer_id` |
| `x04` | 3 / 3 | Which products have no supplier on record? | `missing product_id, sku, category_id, supplier_id, unit_cost, weight_kg, is_discontinued, created_at` |
| `x08` | 3 / 3 | Which products are below their reorder level in any warehouse? | `swapped warehouse_name, reorder_level -> warehouse_id` |
| `m14` | 2 / 3 | What is the average product price in each category? | `swapped category_id -> category_name` |
| `h12` | 2 / 3 | Which products have never been ordered? | `swapped product_id -> unit_price` |
| `h18` | 2 / 3 | How many shipments does each warehouse send? | `swapped warehouse_name -> warehouse_id` |
| `v26` | 2 / 3 | Calculate customer growth month by month. | `missing cumulative_customers` |
| `v28` | 2 / 3 | For each warehouse, how many shipments were delivered versus lost? | `swapped warehouse_name -> warehouse_id` |
| `x02` | 1 / 3 | List orders where no employee handled the sale. | `missing employee_id, required_date, status, total_amount, shipping_cost, currency, created_at` |
| `x07` | 1 / 3 | Which orders have shipped but not yet been delivered? | `swapped shipment_id, warehouse_id, carrier, tracking_number, status -> order_date` |

## Real logic errors: what kind

Tagged by comparing the shape of gold and predicted SQL with SQLGlot. The first structural difference found is the tag, in this priority: wrong tables, join count, GROUP BY / HAVING, window function, LIMIT, NULL handling, DISTINCT, and finally `filter_logic` when the shape matches and only the predicate differs.

| tag | n | easy | medium | hard | very_hard | enterprise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `wrong_tables` | 43 |  |  | 29 | 12 | 2 |
| `filter_logic` | 18 |  |  | 3 | 2 | 13 |
| `aggregation_structure` | 15 |  |  |  |  | 15 |
| `computed column not returned: revenue` | 6 |  |  | 6 |  |  |
| `join_structure` | 4 |  |  | 1 |  | 3 |
| `distinct` | 2 |  |  | 2 |  |  |
| `null_handling` | 2 |  |  |  |  | 2 |
| `computed column not returned: avg_days` | 1 |  |  |  | 1 |  |

By template. Like the projection failures, these cluster: a template either fails wholesale or not at all, which points at a *rule* the model does not know rather than at noise.

| template | tier | failed / total | example question | tag |
| --- | ---: | ---: | ---: | ---: |
| `h02` | hard | 33 / 33 | Who are the top 15 customers by revenue? | `wrong_tables` |
| `x14` | enterprise | 19 / 24 | Which customers have not ordered in the last 26 months? | `aggregation_structure` |
| `v02` | very_hard | 5 / 12 | For every category, list its 2 best selling products. | `wrong_tables` |
| `v11` | very_hard | 3 / 3 | Show the reporting depth of every employee in the org chart. | `wrong_tables` |
| `x10` | enterprise | 3 / 3 | Which orders are still awaiting payment? | `join_structure` |
| `x19` | enterprise | 3 / 3 | What proportion of order value has actually been collected? | `aggregation_structure` |
| `v29` | very_hard | 3 / 3 | Which countries have an above-average number of orders per customer? | `wrong_tables` |
| `x31` | enterprise | 3 / 3 | Which categories have the most stock sitting below reorder level? | `filter_logic` |
| `h08` | hard | 2 / 3 | List employees with the name of the person they report to. | `filter_logic` |
| `h19` | hard | 2 / 33 | List products held at Sao Paulo. | `distinct` |
| `x07` | enterprise | 2 / 3 | Show shipments still in transit. | `filter_logic` |
| `x15` | enterprise | 2 / 3 | Show warehouse utilisation as a percentage of capacity. | `null_handling` |
| `x21` | enterprise | 2 / 3 | Show orders that were paid but never dispatched. | `join_structure` |

(templates with a single failure omitted)

## One example per bucket

### `projection_only` — swapped category_id -> category_name

**What is the average product price in each category?**  (`bench-000790`, medium)

```sql
-- gold
SELECT category_id, AVG(unit_price) AS average_price FROM products GROUP BY category_id ORDER BY average_price DESC
-- predicted
SELECT c.category_name, AVG(p.unit_price) AS average_price FROM categories c JOIN products p ON p.category_id = c.category_id GROUP BY c.category_id, c.category_name ORDER BY average_price DESC
```

rows: gold 110, predicted 110; columns: gold `category_id, average_price`, predicted `category_name, average_price`

### `wrong_rows` — wrong_tables

**Who are the top 15 customers by revenue?**  (`bench-000877`, hard)

```sql
-- gold
SELECT c.customer_name, SUM(o.total_amount) AS revenue FROM customers c JOIN orders o ON o.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name ORDER BY revenue DESC LIMIT 15
-- predicted
SELECT c.customer_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue FROM customers c JOIN orders o ON o.customer_id = c.customer_id JOIN order_items oi ON oi.order_id = o.order_id GROUP BY c.customer_id, c.customer_name ORDER BY revenue DESC LIMIT 15
```

rows: gold 15, predicted 15; columns: gold `customer_name, revenue`, predicted `customer_name, revenue`

### `wrong_values` — computed column not returned: revenue

**Show the 15 highest spending customers.**  (`bench-000878`, hard)

```sql
-- gold
SELECT c.customer_name, SUM(o.total_amount) AS revenue FROM customers c JOIN orders o ON o.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name ORDER BY revenue DESC LIMIT 15
-- predicted
SELECT c.customer_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS total_spent FROM customers c JOIN orders o ON o.customer_id = c.customer_id JOIN order_items oi ON oi.order_id = o.order_id GROUP BY c.customer_id, c.customer_name ORDER BY total_spent DESC LIMIT 15
```

rows: gold 15, predicted 15; columns: gold `customer_name, revenue`, predicted `customer_name, total_spent`

### `execution_error` — Invalid expression / Unexpected token. Line 1, Col: 414.

**Show the top 2 products by revenue within each category.**  (`bench-001648`, very_hard)

```sql
-- gold
WITH ranked AS ( SELECT p.category_id, p.product_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue, ROW_NUMBER() OVER ( PARTITION BY p.category_id ORDER BY SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) DESC ) AS rn FROM products p JOIN order_items oi ON oi.product_id = p.product_id GROUP BY p.category_id, p.product_id, p.product_name) SELECT category_id, product_name, revenue FROM ranked WHERE rn <= 2 ORDER BY category_id, revenue DESC
-- predicted
WITH ranked AS ( SELECT c.category_name, p.product_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue FROM categories c JOIN products p ON p.category_id = c.category_id JOIN order_items oi ON oi.product_id = p.product_id GROUP BY c.category_id, c.category_name, p.product_id, p.product_name) SELECT category_name, product_name, revenue FROM ranked ORDER BY category_name, revenue DESC) WHERE row_number() OVER (PARTITION BY category_name ORDER BY revenue DESC) <= 2 ORDER BY category_name, revenue DESC
```

### `syntax_error` — syntax_error

**What are the 4 highest earning products per category?**  (`bench-001659`, very_hard)

```sql
-- gold
WITH ranked AS ( SELECT p.category_id, p.product_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue, ROW_NUMBER() OVER ( PARTITION BY p.category_id ORDER BY SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) DESC ) AS rn FROM products p JOIN order_items oi ON oi.product_id = p.product_id GROUP BY p.category_id, p.product_id, p.product_name) SELECT category_id, product_name, revenue FROM ranked WHERE rn <= 4 ORDER BY category_id, revenue DESC
-- predicted
WITH ranked AS ( SELECT c.category_name, p.product_name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue FROM categories c JOIN products p ON p.category_id = c.category_id JOIN order_items oi ON oi.product_id = p.product_id GROUP BY c.category_id, p.product_id, p.product_name, c.category_name ORDER BY c.category_name, revenue DESC) SELECT category_name, product_name, revenue FROM ranked ORDER BY category_name, revenue DESC WHERE revenue IN ( SELECT MAX(revenue) FROM ranked WHERE category_name = ranked.category_name) ORDER BY category_name
```

### `schema_hallucination` — unknown_column

**Which orders were shipped after their required date?**  (`bench-001795`, very_hard)

```sql
-- gold
SELECT o.order_id, o.required_date, s.shipped_date FROM orders o JOIN shipments s ON s.order_id = o.order_id WHERE s.shipped_date IS NOT NULL AND s.shipped_date::date > o.required_date ORDER BY o.order_id
-- predicted
SELECT * FROM orders WHERE shipped_date IS NOT NULL AND shipped_date > required_date ORDER BY shipped_date
```

