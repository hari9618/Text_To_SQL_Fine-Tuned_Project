# SFT record format — one worked example

Produced by `scripts/prepare_sft_dataset.py`. This is exactly what the
model sees during Phase 9 training: three chat turns, nothing else.
The `meta` block below sits *outside* `messages` and is never rendered.

- prompt template: `v1` (fingerprint `8288e41a496531a9`) — identical to the frozen baseline
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
SELECT customer_id, customer_name, country FROM customers WHERE country = 'Germany'
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
  "gold_fingerprint": "05b8b14f131d63d42659ffb6daf280a5",
  "gold_row_count": 997
}
```

Carried for offline analysis only — per-difficulty training curves, error slicing. None of it reaches the prompt; `scripts/prepare_sft_dataset.py` asserts that mechanically.