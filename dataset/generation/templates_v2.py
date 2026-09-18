"""Benchmark v2: the v1 templates with the conventions made consistent.

Phase 15 (post-ablation iteration).

The failure analysis of the fine-tuned model (`experiments/FAILURE_ANALYSIS.md`)
found that most remaining failures were not SQL the model could not write but
*conventions it could not know*: which columns a "show orders" question wants,
whether "revenue by customer" means order totals or line items, whether a month
is a date or a number. The v1 templates disagree with each other on each of
those, so no model can learn a rule for them — there is none.

v2 fixes the benchmark, not the metric. `templates.py` (v1) is untouched, so
the frozen v1 numbers stay reproducible. This module applies a documented set
of overrides on top of it and adds a few train-only templates. Every change is
listed in ``CHANGELOG`` with its reason, because a benchmark revision is only
honest if the diff can be read.

Rules that v2 makes consistent, in the order they matter:

1. **Rows without named columns → the whole row.** Forty-one single-table
   "show / list" templates picked their SELECT columns arbitrarily (the two
   *payments* templates in train pick different columns). v2 gold is
   ``SELECT *`` for all of them, and the v2 prompt states the convention. The
   easy tier then measures filtering and ordering — its actual definition —
   rather than column guessing.
2. **Money.** Revenue by customer, country, rep or period is
   ``SUM(orders.total_amount)``; revenue by product, category or supplier is
   line items × (1 − discount). Every train and validation template already
   follows this; one test template (``h02``) did not.
3. **Months are dates.** Three train templates bucket by
   ``DATE_TRUNC('month', ...)``; one test template used ``EXTRACT(MONTH)``.
4. **A paraphrase must not collide with a column name.** ``customers`` has an
   ``is_active`` flag, so "customers inactive for N months" has two defensible
   readings. Reworded to say what it means.
5. **Train covers every SQL construct the benchmark tests.** No v1 training
   example contains ``PARTITION BY`` or ``EXTRACT(EPOCH ...)`` day arithmetic,
   yet both appear in validation and test. Five train-only templates add them
   on different tables and questions than the held-out ones.

The split is **pinned to v1**: every v1 template keeps its v1 split, and new
templates go to train. The v2 test set therefore covers exactly the 48
templates the v1 test set does, with the same questions except where a
paraphrase was reworded (rule 4).
"""

from __future__ import annotations

import re
from dataclasses import replace

from dataset.generation.templates import ALL_TEMPLATES as V1_TEMPLATES
from dataset.generation.templates import T, Template, validate_all as _validate_v1

BENCHMARK_VERSION = "v2"

# ---------------------------------------------------------------------------
# Rule 1: single-table list templates -> SELECT *
# ---------------------------------------------------------------------------

WHOLE_ROW_TEMPLATES: tuple[str, ...] = (
    # customers
    "e01", "e02", "e03", "e04", "e05", "e24", "x06",
    # products
    "e06", "e07", "e08", "e09", "e10", "e11", "x04", "x26",
    # orders
    "e12", "e13", "e14", "e15", "e34", "x02",
    # employees
    "e16", "e17", "e18", "e29", "e30",
    # suppliers / warehouses
    "e19", "e20", "e31", "e32", "x05",
    # payments / shipments / inventory
    "e21", "e25", "e26", "e27", "e22", "e23", "e28", "x07", "e33",
)

_SELECT_LIST = re.compile(r"^SELECT\s+.+?\s+FROM\s", re.IGNORECASE | re.DOTALL)


def _whole_row(sql: str) -> str:
    """Replace the SELECT list with ``*``; everything after FROM is kept."""
    new, n = _SELECT_LIST.subn("SELECT * FROM ", sql, count=1)
    assert n == 1, f"could not rewrite SELECT list: {sql[:80]}"
    return new


# ---------------------------------------------------------------------------
# Rules 2-4: explicit overrides, one per template
# ---------------------------------------------------------------------------

SQL_OVERRIDES: dict[str, str] = {
    # Rule 2. Customer revenue is order totals everywhere else in the
    # benchmark (m11, h13, v01, v20, x16); h02 was the one template using
    # line items for a per-customer figure.
    "h02": """SELECT c.customer_name, SUM(o.total_amount) AS revenue
              FROM customers c JOIN orders o ON o.customer_id = c.customer_id
              GROUP BY c.customer_id, c.customer_name
              ORDER BY revenue DESC LIMIT {n}""",
    # Rule 3. Months are DATE_TRUNC buckets, as in v03, v04, v21.
    "m09": """SELECT DATE_TRUNC('month', order_date) AS month, COUNT(*) AS order_count
              FROM orders WHERE EXTRACT(YEAR FROM order_date) = {year}
              GROUP BY month ORDER BY month""",
}

QUESTION_OVERRIDES: dict[str, tuple[str, ...]] = {
    # Rule 4. "inactive" collides with customers.is_active.
    "x14": ("Which customers have not ordered in the last {churn_months} months?",
            "Show customers with no orders in the last {churn_months} months.",
            "List churned customers with no order in {churn_months} months."),
}

# ---------------------------------------------------------------------------
# Rule 5: train-only templates for constructs absent from v1 training data
# ---------------------------------------------------------------------------

NEW_TRAIN_TEMPLATES: list[Template] = [
    T("n01", "very_hard", "hr", "window partition row_number join",
      ["Show the {small_n} highest paid employees in each department.",
       "For every department, list its {small_n} top earners.",
       "Who are the {small_n} best paid people per department?"],
      """WITH ranked AS (
             SELECT d.department_name, e.first_name, e.last_name, e.salary,
                    ROW_NUMBER() OVER (PARTITION BY e.department_id
                                       ORDER BY e.salary DESC) AS rn
             FROM employees e JOIN departments d ON d.department_id = e.department_id)
         SELECT department_name, first_name, last_name, salary
         FROM ranked WHERE rn <= {small_n}
         ORDER BY department_name, salary DESC"""),

    T("n02", "very_hard", "logistics", "window partition row_number date",
      ["For each carrier, show the {small_n} most recent shipments.",
       "List the {small_n} latest shipments per carrier.",
       "Show every carrier's {small_n} most recently dispatched shipments."],
      """WITH ranked AS (
             SELECT carrier, shipment_id, order_id, shipped_date,
                    ROW_NUMBER() OVER (PARTITION BY carrier
                                       ORDER BY shipped_date DESC) AS rn
             FROM shipments WHERE shipped_date IS NOT NULL)
         SELECT carrier, shipment_id, order_id, shipped_date
         FROM ranked WHERE rn <= {small_n}
         ORDER BY carrier, shipped_date DESC"""),

    T("n03", "very_hard", "sales", "window partition row_number join",
      ["Show the {small_n} largest orders in each customer segment.",
       "For every customer segment, list its {small_n} biggest orders by value.",
       "What are the {small_n} highest value orders per segment?"],
      """WITH ranked AS (
             SELECT c.customer_segment, o.order_id, o.total_amount,
                    ROW_NUMBER() OVER (PARTITION BY c.customer_segment
                                       ORDER BY o.total_amount DESC) AS rn
             FROM orders o JOIN customers c ON c.customer_id = o.customer_id)
         SELECT customer_segment, order_id, total_amount
         FROM ranked WHERE rn <= {small_n}
         ORDER BY customer_segment, total_amount DESC"""),

    T("n04", "very_hard", "logistics", "date_diff avg group_by join",
      ["What is the average number of days from order to dispatch for each warehouse?",
       "How long does each warehouse take to ship an order, on average?",
       "Show mean days between order date and shipped date per warehouse."],
      """SELECT w.warehouse_name,
                AVG(EXTRACT(EPOCH FROM (s.shipped_date - o.order_date)) / 86400.0)
                    AS avg_days_to_ship
         FROM shipments s
         JOIN orders o ON o.order_id = s.order_id
         JOIN warehouses w ON w.warehouse_id = s.warehouse_id
         WHERE s.shipped_date IS NOT NULL
         GROUP BY w.warehouse_id, w.warehouse_name
         ORDER BY avg_days_to_ship"""),

    T("n05", "very_hard", "sales", "date_diff avg cte group_by",
      ["How many days on average pass between signup and a customer's first order, by segment?",
       "For each customer segment, show the mean days from signup to first order.",
       "Average time from signing up to first purchase, per segment."],
      """WITH first_orders AS (
             SELECT customer_id, MIN(order_date) AS first_order
             FROM orders GROUP BY customer_id)
         SELECT c.customer_segment,
                AVG(EXTRACT(EPOCH FROM (f.first_order - c.signup_date::timestamptz)) / 86400.0)
                    AS avg_days_to_first_order
         FROM customers c JOIN first_orders f ON f.customer_id = c.customer_id
         GROUP BY c.customer_segment
         ORDER BY c.customer_segment"""),
]

# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

CHANGELOG: list[tuple[str, str]] = []


def _build() -> list[Template]:
    out: list[Template] = []
    v1_ids = {t.template_id for t in V1_TEMPLATES}
    for t in V1_TEMPLATES:
        tid = t.template_id
        if tid in WHOLE_ROW_TEMPLATES:
            t = replace(t, sql=_whole_row(t.sql))
            CHANGELOG.append((tid, "rule 1: no columns named -> SELECT *"))
        if tid in SQL_OVERRIDES:
            t = replace(t, sql=" ".join(SQL_OVERRIDES[tid].split()))
            CHANGELOG.append((tid, "rule 2/3: gold rewritten to the benchmark's own convention"))
        if tid in QUESTION_OVERRIDES:
            t = replace(t, questions=QUESTION_OVERRIDES[tid])
            CHANGELOG.append((tid, "rule 4: paraphrase collided with a column name"))
        out.append(t)
    for t in NEW_TRAIN_TEMPLATES:
        assert t.template_id not in v1_ids, f"{t.template_id} collides with v1"
        out.append(t)
        CHANGELOG.append((t.template_id, "rule 5: new train-only template"))
    return out


ALL_TEMPLATES: list[Template] = _build()
NEW_TEMPLATE_IDS: frozenset[str] = frozenset(t.template_id for t in NEW_TRAIN_TEMPLATES)


def validate_all() -> None:
    """v1's own checks, run against the v2 list."""
    _validate_v1(ALL_TEMPLATES)


def changed_template_ids() -> set[str]:
    return {tid for tid, _ in CHANGELOG}
