# SFT record format — one worked example

Produced by `scripts/prepare_sft_dataset.py`. This is exactly what the
model sees during Phase 9 training: three chat turns, nothing else.
The `meta` block below sits *outside* `messages` and is never rendered.

- prompt template: `v2` (fingerprint `4e72cc5f722ce436`) — from src.model.prompt_v2
- schema: full, 5,455 chars, fingerprint `d03619e711661bc5`

---

## 1. system

```text
You are an expert PostgreSQL analyst. You convert business questions into correct, executable PostgreSQL queries.

Rules:
- Output ONLY the SQL query. No explanation, no commentary.
- Use only the tables and columns given in the schema.
- Write a single SELECT statement. Never modify data.
- Use PostgreSQL syntax.
- When a foreign key is nullable, consider whether LEFT JOIN is needed to avoid silently dropping rows.

Reference date:
- "Today" for this database is TIMESTAMPTZ '2026-08-01 00:00:00+00:00'. Use that literal for "now", "today", "this year", "last N days/months" and similar. Never use NOW() or CURRENT_DATE.

Business definitions:
- Order value, amount spent, and revenue by customer, country, sales rep, department, segment or time period = SUM(orders.total_amount).
- Revenue by product, category or supplier = SUM(order_items.quantity * order_items.unit_price * (1 - order_items.discount_pct / 100)).
- Net or actual revenue excludes orders whose status is 'cancelled' or 'returned'.
- A payment counts only when payments.status = 'completed'. An order is awaiting payment when it is not cancelled and has no completed payment.
- A shipment is delivered when delivered_date IS NOT NULL; in transit when shipped_date IS NOT NULL and delivered_date IS NULL.

Output conventions:
- If the question asks to show or list records without naming columns, return every column of that table (SELECT *).
- Identify customers, products, suppliers, warehouses, categories and departments by their name column when that table is in the query; employees by first_name and last_name.
- Group by calendar month with DATE_TRUNC('month', <date>) and by year with EXTRACT(YEAR FROM <date>).
- Days between two timestamps: EXTRACT(EPOCH FROM (later - earlier)) / 86400.0.
```

## 2. user

The full schema is embedded here. Truncated below for readability —
the real record contains all 5,455 characters.

```text
Database schema:

CREATE TABLE categories (
  category_id INTEGER NOT NULL PRIMARY KEY,
  category_name VARCHAR(100) NOT NULL,
  parent_category_id INTEGER  -- Parent category, NULL for top-level categories. Self-referencing hierarchy.,
  description TEXT,
  created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE customers (
  customer_id INTEGER NOT NULL PRIMARY KEY,
  customer_name VARCHAR(200) NOT NULL,
  email VARCHAR(255) NOT NULL,
  phone VARCHAR(40),
  country VARCHAR(100) NOT NULL,
  city VARCHAR(100),
  customer_segment VARCHAR(20) NOT NULL  -- Either individual or business. Business customers may have a credit_limit.,
  credit_limit NUMERIC(12,2)  -- Maximum outstanding balance allowed. N

        ... schema continues ...

  products.category_id -> categories.category_id
--   products.supplier_id -> suppliers.supplier_id  [nullable: use LEFT JOIN to keep unmatched rows]
--   shipments.order_id -> orders.order_id
--   shipments.warehouse_id -> warehouses.warehouse_id

Question: Show all customers from Germany.

PostgreSQL query: /no_think
```

## 3. assistant

SQL only — no prose, no markdown fence, no trailing semicolon.

```sql
SELECT * FROM customers WHERE country = 'Germany'
```

---

## meta (not shown to the model)

```json
{
  "template_id": "e01",
  "phrasing_index": 0,
  "slot_values": {
    "country": "Germany"
  },
  "referenced_tables": [
    "customers"
  ],
  "split": "train",
  "difficulty": "easy",
  "domain": "sales",
  "query_type": [
    "select",
    "where"
  ],
  "gold_fingerprint": "189fde62c5689dfd13214cc4c0d9fdea",
  "gold_row_count": 997
}
```

Carried for offline analysis only — per-difficulty training curves, error slicing. None of it reaches the prompt; `scripts/prepare_sft_dataset.py` asserts that mechanically.