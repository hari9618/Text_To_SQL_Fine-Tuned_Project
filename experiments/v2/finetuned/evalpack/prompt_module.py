"""Prompt v2: v1 plus the conventions the failure analysis showed were missing.

Phase 15.

v1 (`src/model/prompt.py`) is deliberately plain and stays frozen — the base
and fine-tuned v1 numbers are measured against it. v2 adds three things the
model cannot infer from the schema alone:

* **A reference date.** Questions like "in the last 6 months" need a "now".
  The benchmark anchors every such question to ``DATA_AS_OF`` rather than
  ``NOW()`` so answers do not change overnight; v1 never told the model that,
  so it either used NOW() or guessed at an ``is_active`` flag.
* **A business glossary.** What "revenue", "spend", "net revenue", "paid" and
  "awaiting payment" mean in this database. Every one of these is a rule the
  training templates already follow; they were simply unstated. A production
  system carries the same thing as a semantic layer.
* **Output conventions.** Whole row when no columns are named; months as
  ``DATE_TRUNC`` buckets; PostgreSQL date arithmetic; entities by name.

Kept import-free on purpose, like ``repair_prompt.py``: this file is shipped
verbatim to the GPU host inside the eval pack and rendered there, so it must
not depend on anything else in ``src``. ``tests/test_prompt_v2.py`` asserts
the literal date below matches ``src.constants.DATA_AS_OF``.
"""

from __future__ import annotations

import hashlib

PROMPT_VERSION = "v2"

# Must equal src.constants.DATA_AS_OF_SQL. Literal here so the module stays
# import-free; the test suite enforces the equality.
REFERENCE_DATE_SQL = "TIMESTAMPTZ '2026-08-01 00:00:00+00:00'"

SYSTEM_PROMPT = f"""You are an expert PostgreSQL analyst. You convert business \
questions into correct, executable PostgreSQL queries.

Rules:
- Output ONLY the SQL query. No explanation, no commentary.
- Use only the tables and columns given in the schema.
- Write a single SELECT statement. Never modify data.
- Use PostgreSQL syntax.
- When a foreign key is nullable, consider whether LEFT JOIN is needed to \
avoid silently dropping rows.

Reference date:
- "Today" for this database is {REFERENCE_DATE_SQL}. Use that literal for \
"now", "today", "this year", "last N days/months" and similar. Never use NOW() \
or CURRENT_DATE.

Business definitions:
- Order value, amount spent, and revenue by customer, country, sales rep, \
department, segment or time period = SUM(orders.total_amount).
- Revenue by product, category or supplier = SUM(order_items.quantity * \
order_items.unit_price * (1 - order_items.discount_pct / 100)).
- Net or actual revenue excludes orders whose status is 'cancelled' or \
'returned'.
- A payment counts only when payments.status = 'completed'. An order is \
awaiting payment when it is not cancelled and has no completed payment.
- A shipment is delivered when delivered_date IS NOT NULL; in transit when \
shipped_date IS NOT NULL and delivered_date IS NULL.

Output conventions:
- If the question asks to show or list records without naming columns, \
return every column of that table (SELECT *).
- Identify customers, products, suppliers, warehouses, categories and \
departments by their name column when that table is in the query; employees \
by first_name and last_name.
- Group by calendar month with DATE_TRUNC('month', <date>) and by year with \
EXTRACT(YEAR FROM <date>).
- Days between two timestamps: EXTRACT(EPOCH FROM (later - earlier)) / 86400.0."""

USER_TEMPLATE = """Database schema:

{schema}

Question: {question}

PostgreSQL query:"""


def build_messages(question: str, schema: str) -> list[dict[str, str]]:
    """Same shape as v1: system + user, ``/no_think`` appended."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(schema=schema, question=question)
                       + " /no_think",
        },
    ]


def prompt_fingerprint() -> str:
    """Hash of the template text, recorded alongside results."""
    combined = SYSTEM_PROMPT + "\x00" + USER_TEMPLATE + "\x00" + PROMPT_VERSION
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
