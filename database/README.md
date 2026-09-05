# `database/`

Everything that defines and populates the PostgreSQL database the Text-to-SQL
system queries.

This directory is **assets and workflow**, not application code — it is executed
with `psql`, never imported by the running service. Python code that *talks to*
the database lives in `src/sql/`.

---

## Files

| File         | Purpose                                                    | Phase |
| ------------ | ---------------------------------------------------------- | ----- |
| `schema.sql` | DDL — tables, keys, constraints, indexes. Structure only.  | 2     |
| `seed.sql`   | Reference/lookup data small enough to hand-write.          | 3     |
| `README.md`  | This file.                                                 | —     |

Bulk synthetic data (thousands of orders, customers, etc.) is **not** written
here by hand. Phase 3 generates it with a Python script in `scripts/`, because
realistic distributions, dates, and foreign-key consistency cannot reasonably be
authored by hand.

`seed.sql` is only for small, stable reference rows — country codes, order
statuses, product categories — the kind of data that is part of the schema's
meaning rather than part of the generated dataset.

---

## Run order

Order matters. `schema.sql` creates the tables that `seed.sql` inserts into.

```powershell
# 1. structure
psql -U postgres -d enterprise_sql -f database/schema.sql

# 2. reference data
psql -U postgres -d enterprise_sql -f database/seed.sql
```

`psql` is not on PATH by default on this machine. Either add
`C:\Program Files\PostgreSQL\18\bin` to PATH, or call it in full:

```powershell
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -d enterprise_sql -f database/schema.sql
```

Run these from the **project root** so the relative paths resolve.

---

## Why the schema is rebuildable from scratch

`schema.sql` is written to be **idempotent** — safe to run repeatedly. During
Phases 2–4 the schema will change often, and every change invalidates parts of
the benchmark dataset built against it.

This matters more than it looks. The evaluation in Phase 10 compares a base
model against a fine-tuned model on the *same* questions against the *same*
database. If the database cannot be reconstructed exactly, the comparison is not
reproducible and the headline result is worthless.

So: **the database is disposable, these files are the source of truth.** Never
fix a problem by editing the live database by hand — change the file and reload.

---

## Conventions

Decided once here so generated SQL, benchmark queries, and the schema all agree.
Consistency is not cosmetic: an inconsistently named schema teaches the
fine-tuned model to guess, which shows up directly as schema hallucination in
the Phase 10 metrics.

- **Table names** — `snake_case`, plural: `customers`, `order_items`
- **Column names** — `snake_case`, singular: `customer_id`, `total_amount`
- **Primary keys** — `<singular_table>_id`, e.g. `customers.customer_id`
- **Foreign keys** — carry the referenced key's exact name, so join columns
  match: `orders.customer_id` → `customers.customer_id`
- **Timestamps** — `TIMESTAMPTZ`, not `TIMESTAMP`. Enterprise queries filter on
  dates constantly; timezone-naive columns produce quietly wrong answers.
- **Money** — `NUMERIC(12, 2)`, never `FLOAT`. Floating point cannot represent
  currency exactly and breaks execution-accuracy comparison on aggregates.
- **Constraints are named explicitly** — `fk_orders_customer`,
  `chk_order_total_non_negative`. Named constraints produce readable PostgreSQL
  errors, which Phase 11's SQL repair loop feeds back to the model. An anonymous
  `$2` in an error message is a much weaker repair signal.
- **Comments** — use `COMMENT ON` for anything ambiguous. Phase 6 schema
  retrieval can surface these to the model as column descriptions.

---

## Credentials

Never hard-coded, never committed. Connection details come from environment
variables via `.env` (git-ignored); `.env.example` holds placeholders only.
