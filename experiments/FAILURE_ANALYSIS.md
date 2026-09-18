# Failure analysis

Configuration: **fine-tuned + repair (configuration 5)** — `experiments/repair/results.jsonl`

453 test questions, **236 correct (52.1 %)**, 217 failures. Every failure was re-executed against PostgreSQL next to its gold query and sorted into one bucket.

> **This is a post-hoc diagnostic on the test set.** It explains the frozen
> number; it does not change it. Any fix it motivates must be chosen on the
> validation split and re-measured as a new configuration.

## Where the failures are

| bucket | n | of failures | of test set | meaning |
| --- | ---: | ---: | ---: | ---: |
| `projection_only` | 118 | 54.4 % | 26.0 % | right rows, different column set — the question never said which columns |
| `wrong_values` | 4 | 1.8 % | 0.9 % | same rows identified, different values — a bad computation |
| `wrong_rows` | 87 | 40.1 % | 19.2 % | genuinely different rows — a real logic error |
| `schema_hallucination` | 3 | 1.4 % | 0.7 % | referenced a table or column that does not exist |
| `execution_error` | 4 | 1.8 % | 0.9 % | ran but PostgreSQL rejected it (type error, ambiguity, ...) |
| `syntax_error` | 1 | 0.5 % | 0.2 % | did not parse |

Read it as three groups:

* **118 benign** (`projection_only` + `column_order`) — the rows are right. The question did not say which columns to return and the model chose a different set from the gold query. 54.4 % of all failures.
* **91 real logic errors** (`wrong_rows` + `wrong_values`) — the model misunderstood the question or the schema. 41.9 % of failures.
* **8 detectable** (hallucination, execution, syntax) — the only failures repair can see. 3.7 % of failures.

### Row-level accuracy (diagnostic, not the headline)

If a question does not specify columns and the model returns the right rows under a different column set, that is arguably correct. Counting `projection_only` and `column_order` as correct gives:

|  | strict (headline) | row-level (diagnostic) |
| --- | ---: | ---: |
| all | 52.1 % | 78.1 % |
| easy | 23.0 % | 99.2 % |
| medium | 60.3 % | 61.9 % |
| hard | 84.7 % | 88.2 % |
| very_hard | 8.3 % | 22.2 % |
| enterprise | 52.4 % | 65.5 % |

The strict number stays the headline: the benchmark was frozen before any of this was looked at, and loosening a metric after seeing test results is exactly the move this project refuses to make. The row-level figure is here so the gap between the two is visible and explained, not hidden.

## By difficulty

| tier | n | correct | `projection_only` | `wrong_values` | `wrong_rows` | `schema_hallucination` | `execution_error` | `syntax_error` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 126 | 29 | 96 |  | 1 |  |  |  |
| medium | 63 | 38 | 1 |  | 24 |  |  |  |
| hard | 144 | 122 | 5 |  | 17 |  |  |  |
| very_hard | 36 | 3 | 5 | 4 | 20 | 1 | 2 | 1 |
| enterprise | 84 | 44 | 11 |  | 25 | 2 | 2 |  |

## `projection_only`: which templates

Every question generated from the same template fails the same way, which is the signature of a convention mismatch rather than a model error — the model cannot know which four columns an unseen template's author chose.

| template | failed / total | example question | difference |
| --- | ---: | ---: | ---: |
| `e25` | 27 / 27 | Show payments larger than 3528 dollars. | `missing payment_method` |
| `e13` | 24 / 24 | Show orders placed in 2023. | `missing total_amount` |
| `e28` | 24 / 24 | Show shipments dispatched in 2019. | `missing carrier` |
| `e22` | 14 / 15 | Show shipments handled by AeroCargo. | `missing carrier` |
| `e33` | 6 / 27 | List stock entries below 360 units. | `missing inventory_id` |
| `v16` | 3 / 3 | Show each customer's first order date and their total order count. | `swapped customer_name -> customer_id` |
| `x02` | 3 / 3 | Which orders have no sales representative assigned? | `swapped order_date, total_amount -> employee_id` |
| `x05` | 3 / 3 | Show suppliers that have not been rated yet. | `missing country` |
| `h12` | 2 / 3 | Which products have never been ordered? | `missing product_id` |
| `h18` | 2 / 3 | How many shipments does each warehouse send? | `swapped warehouse_name -> warehouse_id` |
| `x04` | 2 / 3 | Which products have no supplier on record? | `swapped product_id -> unit_price` |
| `x08` | 2 / 3 | Which products are below their reorder level in any warehouse? | `swapped warehouse_name, reorder_level -> unit_price, warehouse_id` |
| `m14` | 1 / 3 | What is the average product price in each category? | `swapped category_id -> category_name` |
| `x07` | 1 / 3 | Which orders have shipped but not yet been delivered? | `swapped shipment_id, carrier -> order_date, delivered_date` |
| `e27` | 1 / 9 | List all completed payments. | `swapped status -> payment_method` |
| `h32` | 1 / 33 | List the 2 warehouses with the highest stock value. | `swapped warehouse_name -> warehouse_id` |
| `v26` | 1 / 3 | Calculate customer growth month by month. | `missing cumulative_customers` |
| `v28` | 1 / 3 | For each warehouse, how many shipments were delivered versus lost? | `swapped warehouse_name -> warehouse_id, total_shipments` |

## Real logic errors: what kind

Tagged by comparing the shape of gold and predicted SQL with SQLGlot. The first structural difference found is the tag, in this priority: wrong tables, join count, GROUP BY / HAVING, window function, LIMIT, NULL handling, DISTINCT, and finally `filter_logic` when the shape matches and only the predicate differs.

| tag | n | easy | medium | hard | very_hard | enterprise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `wrong_tables` | 44 | 1 |  | 14 | 17 | 12 |
| `filter_logic` | 33 |  | 24 | 2 | 2 | 5 |
| `join_structure` | 3 |  |  | 1 |  | 2 |
| `null_handling` | 3 |  |  |  |  | 3 |
| `aggregation_structure` | 3 |  |  |  |  | 3 |
| `computed column not returned: avg_days` | 2 |  |  |  | 2 |  |
| `computed column not returned: level` | 1 |  |  |  | 1 |  |
| `window_function` | 1 |  |  |  | 1 |  |
| `computed column not returned: lost` | 1 |  |  |  | 1 |  |

By template. Like the projection failures, these cluster: a template either fails wholesale or not at all, which points at a *rule* the model does not know rather than at noise.

| template | tier | failed / total | example question | tag |
| --- | ---: | ---: | ---: | ---: |
| `m09` | medium | 24 / 24 | How many orders were placed each month of 2025? | `filter_logic` |
| `v02` | very_hard | 12 / 12 | Show the top 2 products by revenue within each category. | `wrong_tables` |
| `h02` | hard | 11 / 33 | Show the 15 highest spending customers. | `wrong_tables` |
| `x14` | enterprise | 8 / 24 | Show customers inactive for over 26 months. | `wrong_tables` |
| `x10` | enterprise | 3 / 3 | Which orders are still awaiting payment? | `join_structure` |
| `x19` | enterprise | 3 / 3 | What proportion of order value has actually been collected? | `aggregation_structure` |
| `v29` | very_hard | 3 / 3 | Which countries have an above-average number of orders per customer? | `wrong_tables` |
| `h05` | hard | 2 / 3 | How many products does each supplier provide? | `wrong_tables` |
| `v08` | very_hard | 2 / 3 | Which carrier delivers fastest on average? | `computed column not returned: avg_days` |
| `v15` | very_hard | 2 / 3 | Show late shipments compared to the required date. | `filter_logic` |
| `x01` | enterprise | 2 / 3 | Show real revenue after removing cancellations and returns. | `filter_logic` |
| `x15` | enterprise | 2 / 3 | How full is each warehouse relative to its capacity? | `null_handling` |
| `v28` | very_hard | 2 / 3 | Show delivered and lost shipment counts per warehouse. | `wrong_tables` |
| `x21` | enterprise | 2 / 3 | Which paid orders have never been shipped? | `wrong_tables` |
| `x31` | enterprise | 2 / 3 | Show understocked categories. | `aggregation_structure` |

(templates with a single failure omitted)

## One example per bucket

### `projection_only` — missing total_amount

**Show orders placed in 2023.**  (`bench-000322`, easy)

```sql
-- gold
SELECT order_id, customer_id, order_date, total_amount FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023
-- predicted
SELECT order_id, customer_id, order_date FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023
```

rows: gold 6783, predicted 6783; columns: gold `order_id, customer_id, order_date, total_amount`, predicted `order_id, customer_id, order_date`

### `wrong_rows` — wrong_tables

**Show shipments handled by GlobalEx.**  (`bench-000565`, easy)

```sql
-- gold
SELECT shipment_id, order_id, carrier, status FROM shipments WHERE carrier = 'GlobalEx'
-- predicted
SELECT s.shipment_id, o.order_id, s.status FROM shipments s JOIN orders o ON o.order_id = s.order_id WHERE o.customer_id IN (SELECT c.customer_id FROM customers c WHERE c.customer_name = 'GlobalEx')
```

rows: gold 8253, predicted 0; columns: gold `shipment_id, order_id, carrier, status`, predicted `shipment_id, order_id, status`

### `wrong_values` — computed column not returned: avg_days

**Which carrier delivers fastest on average?**  (`bench-001717`, very_hard)

```sql
-- gold
SELECT s.carrier, AVG(EXTRACT(EPOCH FROM (s.delivered_date - s.shipped_date)) / 86400.0) AS avg_days FROM shipments s WHERE s.delivered_date IS NOT NULL AND s.shipped_date IS NOT NULL GROUP BY s.carrier ORDER BY avg_days ASC
-- predicted
SELECT carrier, AVG(EXTRACT(DAY FROM (delivered_date - shipped_date))) AS average_days FROM shipments GROUP BY carrier ORDER BY average_days
```

rows: gold 5, predicted 5; columns: gold `carrier, avg_days`, predicted `carrier, average_days`

### `execution_error` — function datediff(unknown, timestamp with time zone, timestamp with time zone) d

**Compare carriers by average delivery time.**  (`bench-001719`, very_hard)

```sql
-- gold
SELECT s.carrier, AVG(EXTRACT(EPOCH FROM (s.delivered_date - s.shipped_date)) / 86400.0) AS avg_days FROM shipments s WHERE s.delivered_date IS NOT NULL AND s.shipped_date IS NOT NULL GROUP BY s.carrier ORDER BY avg_days ASC
-- predicted
SELECT s.carrier, AVG(DATEDIFF('day', o.order_date, s.delivered_date)) AS average_days FROM orders o JOIN shipments s ON s.order_id = o.order_id GROUP BY s.carrier ORDER BY average_days DESC
```

### `schema_hallucination` — unknown_column

**Which orders were shipped after their required date?**  (`bench-001795`, very_hard)

```sql
-- gold
SELECT o.order_id, o.required_date, s.shipped_date FROM orders o JOIN shipments s ON s.order_id = o.order_id WHERE s.shipped_date IS NOT NULL AND s.shipped_date::date > o.required_date ORDER BY o.order_id
-- predicted
SELECT order_id, customer_id, required_date, shipped_date FROM orders WHERE shipped_date IS NOT NULL AND shipped_date > required_date
```

### `syntax_error` — syntax_error

**Calculate mean days from order to payment.**  (`bench-002909`, very_hard)

```sql
-- gold
SELECT AVG(EXTRACT(EPOCH FROM (p.payment_date - o.order_date)) / 86400.0) AS avg_days_to_payment FROM payments p JOIN orders o ON o.order_id = p.order_id WHERE p.status = 'completed'
-- predicted
SELECT AVG(DATEDIFF(payment.payment_date, order.order_date)) AS average_days FROM orders JOIN payments payment ON payment.order_id = orders.order_id
```

