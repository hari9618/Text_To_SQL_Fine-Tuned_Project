"""Slot values sampled from the live database.

Phase 2, revised in Phase 3.

Templates are parameterised: "Show all customers from {country}". Filling a slot
with a value that actually exists matters for two reasons.

First, the resulting SQL returns rows. A question whose correct answer is an
empty result set cannot distinguish a correct query from a subtly wrong one —
both return nothing, and both would score as correct.

Second, it keeps the benchmark grounded. Inventing 'Wakanda' would teach the
model that plausible-sounding literals are acceptable, which is exactly the
hallucination behaviour this project exists to measure.

Numeric thresholds are derived the same way, and for the same reasons. The
first version drew them from hand-written lists — "salary above 40000",
"credit limit below 1000" — chosen without knowing the data. Two failure modes
followed:

*Empty results.* The minimum credit limit is 5,000, so "below 1,000" matched
nothing. Phase 3 rejected 81 examples for this.

*Vacuous predicates.* The minimum salary exceeds 50,000, so "salary > 40000"
and "salary > 50000" both return every employee. Identical result sets mean a
model that ignores the threshold entirely still scores correct — the question
tests nothing.

Both are fixed by taking thresholds from **interior quantiles of the column the
threshold is compared against**. A value strictly between the minimum and the
maximum always splits the rows into two non-empty parts, so the predicate
always does real work.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.constants import DATA_AS_OF_SQL  # noqa: E402

# Deciles. Interior only — 0.0 and 1.0 would reproduce the min and max, which
# are exactly the values that make a predicate vacuous or empty.
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class ValuePools:
    """Slot name -> list of candidate values."""

    pools: dict[str, list[Any]] = field(default_factory=dict)

    def __getitem__(self, slot: str) -> list[Any]:
        if slot not in self.pools:
            raise KeyError(
                f"No value pool for slot {slot!r}. "
                f"Known slots: {sorted(self.pools)}"
            )
        return self.pools[slot]

    def __contains__(self, slot: str) -> bool:
        return slot in self.pools


# ---------------------------------------------------------------------------
# Structural pools
#
# These control query *shape*, not row selection: a LIMIT of 5 versus 10, or
# NTILE(4) versus NTILE(3). They cannot produce an empty or vacuous result, so
# there is nothing to derive from the data.
# ---------------------------------------------------------------------------

STRUCTURAL_POOLS: dict[str, list[Any]] = {
    "n": [3, 5, 8, 10, 12, 15, 20, 25, 30, 50, 100],
    "small_n": [2, 3, 4, 5],
    "month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "quarter": [1, 2, 3, 4],
}

# Calendar pools come from the data's actual range, so no question can ask
# about a year with no orders in it.
#
# `year` is used for equality ("orders placed in 2023"), so every year present
# in the data is valid. `signup_year` and `hire_year` are used with `>`
# ("signed up after 2024"), so they must exclude the latest year — otherwise
# the predicate matches nothing.
CALENDAR_POOL_QUERIES: dict[str, str] = {
    "year": (
        "SELECT DISTINCT EXTRACT(YEAR FROM order_date)::int FROM orders "
        "ORDER BY 1"
    ),
    "signup_year": (
        "SELECT DISTINCT EXTRACT(YEAR FROM signup_date)::int AS y "
        "FROM customers "
        "WHERE EXTRACT(YEAR FROM signup_date) < "
        "      (SELECT MAX(EXTRACT(YEAR FROM signup_date)) FROM customers) "
        "ORDER BY y"
    ),
    "hire_year": (
        "SELECT DISTINCT EXTRACT(YEAR FROM hire_date)::int AS y "
        "FROM employees "
        "WHERE EXTRACT(YEAR FROM hire_date) < "
        "      (SELECT MAX(EXTRACT(YEAR FROM hire_date)) FROM employees) "
        "ORDER BY y"
    ),
    # A LIMIT larger than the number of rows available is vacuous: "top 50
    # warehouses" and "top 100 warehouses" both return all 13. Bounded by the
    # count of warehouses that actually hold stock.
    "warehouse_limit": (
        "SELECT generate_series(2, GREATEST("
        "  (SELECT COUNT(DISTINCT warehouse_id) FROM inventory) - 1, 2))"
    ),
}


# ---------------------------------------------------------------------------
# Categorical pools — real values from real columns
# ---------------------------------------------------------------------------

CATEGORICAL_POOL_QUERIES: dict[str, str] = {
    "country": "SELECT DISTINCT country FROM customers ORDER BY country",
    "city": (
        "SELECT city FROM customers WHERE city IS NOT NULL "
        "GROUP BY city ORDER BY COUNT(*) DESC LIMIT 25"
    ),
    "supplier_country": "SELECT DISTINCT country FROM suppliers ORDER BY country",
    "warehouse_country": "SELECT DISTINCT country FROM warehouses ORDER BY country",
    # Only cities whose warehouses actually hold stock. Two warehouses are
    # cross-dock facilities with no inventory, and asking what is stored there
    # correctly returns nothing.
    "warehouse_city": (
        "SELECT DISTINCT w.city FROM warehouses w "
        "JOIN inventory i ON i.warehouse_id = w.warehouse_id ORDER BY 1"
    ),
    "top_category": (
        "SELECT category_name FROM categories "
        "WHERE parent_category_id IS NULL ORDER BY category_name"
    ),
    # Only categories that actually hold products: the empty ones would make
    # "show products in X" return nothing.
    "category": (
        "SELECT c.category_name FROM categories c "
        "JOIN products p ON p.category_id = c.category_id "
        "GROUP BY c.category_name ORDER BY COUNT(*) DESC LIMIT 40"
    ),
    "segment": "SELECT DISTINCT customer_segment FROM customers ORDER BY 1",
    "status": "SELECT DISTINCT status FROM orders ORDER BY 1",
    "ship_status": "SELECT DISTINCT status FROM shipments ORDER BY 1",
    "pay_method": "SELECT DISTINCT payment_method FROM payments ORDER BY 1",
    "pay_status": "SELECT DISTINCT status FROM payments ORDER BY 1",
    "carrier": (
        "SELECT DISTINCT carrier FROM shipments WHERE carrier IS NOT NULL "
        "ORDER BY 1"
    ),
    "department": "SELECT department_name FROM departments ORDER BY department_name",
    "job_title": "SELECT DISTINCT job_title FROM employees ORDER BY 1",
    "warehouse": "SELECT warehouse_name FROM warehouses ORDER BY warehouse_name",
}


# ---------------------------------------------------------------------------
# Derived numeric pools
#
# Each entry names the expression a threshold will be compared against. The
# slot name says which column it belongs to, so two templates comparing against
# different distributions never share a pool — the mistake that produced
# "average salary above 100000" when no department comes close.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NumericPool:
    """A threshold pool derived from a distribution in the database.

    Args:
        source: query returning a single numeric column aliased ``x``.
        digits: decimal places to round candidates to. 0 yields ints, which
            read naturally in a question ("above 250" not "above 250.0").
        discrete: True for small-integer distributions such as counts per
            group, where every interior value is worth using and quantiles
            would collapse onto the same few numbers.
    """

    source: str
    digits: int = 0
    discrete: bool = False


NUMERIC_POOLS: dict[str, NumericPool] = {
    # --- catalogue ---
    "product_price": NumericPool("SELECT unit_price AS x FROM products"),
    "category_avg_price": NumericPool(
        "SELECT AVG(unit_price) AS x FROM products GROUP BY category_id"),
    "product_weight": NumericPool(
        "SELECT weight_kg AS x FROM products WHERE weight_kg IS NOT NULL",
        digits=1),
    "supplier_rating": NumericPool(
        "SELECT rating AS x FROM suppliers WHERE rating IS NOT NULL", digits=1),

    # --- sales ---
    "order_total": NumericPool("SELECT total_amount AS x FROM orders"),
    "order_shipping": NumericPool("SELECT shipping_cost AS x FROM orders"),
    "customer_spend": NumericPool(
        "SELECT SUM(total_amount) AS x FROM orders GROUP BY customer_id"),
    "customer_order_count": NumericPool(
        "SELECT COUNT(*) AS x FROM orders GROUP BY customer_id", discrete=True),
    "customer_category_count": NumericPool(
        "SELECT COUNT(DISTINCT p.category_id) AS x FROM orders o "
        "JOIN order_items oi ON oi.order_id = o.order_id "
        "JOIN products p ON p.product_id = oi.product_id "
        "GROUP BY o.customer_id", discrete=True),
    "country_customer_count": NumericPool(
        "SELECT COUNT(*) AS x FROM customers GROUP BY country", discrete=True),
    "order_line_count": NumericPool(
        "SELECT COUNT(*) AS x FROM order_items GROUP BY order_id", discrete=True),
    "product_total_qty": NumericPool(
        "SELECT SUM(quantity) AS x FROM order_items GROUP BY product_id"),
    "customer_credit_limit": NumericPool(
        "SELECT credit_limit AS x FROM customers WHERE credit_limit IS NOT NULL"),

    # --- hr ---
    "employee_salary": NumericPool("SELECT salary AS x FROM employees"),
    "dept_avg_salary": NumericPool(
        "SELECT AVG(salary) AS x FROM employees GROUP BY department_id"),

    # --- finance ---
    "payment_amount": NumericPool("SELECT amount AS x FROM payments"),
    "payment_method_uses": NumericPool(
        "SELECT COUNT(*) AS x FROM payments GROUP BY payment_method",
        discrete=True),

    # --- logistics ---
    "inventory_qty": NumericPool(
        "SELECT quantity_on_hand AS x FROM inventory"),
    "product_total_stock": NumericPool(
        "SELECT SUM(quantity_on_hand) AS x FROM inventory GROUP BY product_id"),
    "product_warehouse_count": NumericPool(
        "SELECT COUNT(DISTINCT warehouse_id) AS x FROM inventory "
        "GROUP BY product_id", discrete=True),

    # --- time windows ---
    # Measured from DATA_AS_OF, the same anchor the templates use. Measuring
    # from NOW() while the queries filter on DATA_AS_OF would compute
    # thresholds against a different reference point than the one they are
    # applied to, drifting further apart every day.
    "pending_days": NumericPool(
        f"SELECT EXTRACT(EPOCH FROM ({DATA_AS_OF_SQL} - order_date)) / 86400.0 AS x "
        "FROM orders WHERE status = 'pending'"),
    "churn_months": NumericPool(
        f"SELECT EXTRACT(EPOCH FROM ({DATA_AS_OF_SQL} - MAX(order_date))) "
        "/ 2592000.0 AS x FROM orders GROUP BY customer_id"),
    "inactive_recent_months": NumericPool(
        f"SELECT EXTRACT(EPOCH FROM ({DATA_AS_OF_SQL} - MAX(o.order_date))) "
        "/ 2592000.0 AS x "
        "FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
        "WHERE c.is_active = FALSE GROUP BY o.customer_id"),
}


def _continuous_query(pool: NumericPool) -> str:
    """Interior quantiles, rounded, with min and max excluded.

    Excluding the bounds is the whole point: a threshold equal to the minimum
    or maximum makes the predicate match everything or nothing.
    """
    quantiles = ", ".join(str(q) for q in QUANTILES)
    return f"""
        WITH src AS ({pool.source}),
             bounds AS (SELECT MIN(x) AS lo, MAX(x) AS hi FROM src),
             q AS (
                 SELECT unnest(
                     percentile_cont(ARRAY[{quantiles}]) WITHIN GROUP (ORDER BY x)
                 ) AS v FROM src
             )
        SELECT DISTINCT ROUND(q.v::numeric, {pool.digits}) AS value
        FROM q, bounds
        WHERE ROUND(q.v::numeric, {pool.digits}) > bounds.lo
          AND ROUND(q.v::numeric, {pool.digits}) < bounds.hi
        ORDER BY value
    """


def _discrete_query(pool: NumericPool) -> str:
    """Every distinct value in [min, max), for small-integer distributions.

    The upper bound is exclusive so that ``> value`` always leaves rows behind,
    and the lower bound is inclusive because ``> min`` still excludes the
    minimum group — a real filter, not a vacuous one.
    """
    return f"""
        WITH src AS ({pool.source}),
             bounds AS (SELECT MIN(x) AS lo, MAX(x) AS hi FROM src)
        SELECT DISTINCT src.x::int AS value
        FROM src, bounds
        WHERE src.x >= bounds.lo AND src.x < bounds.hi
        ORDER BY value
    """


def _coerce(value: Any, digits: int) -> Any:
    """Render a Decimal as the kind of number a person would write."""
    if isinstance(value, Decimal):
        return int(value) if digits == 0 else float(value)
    return value


def load_value_pools(conn: psycopg.Connection) -> ValuePools:
    """Read every pool from the database, then merge in the structural ones."""
    pools: dict[str, list[Any]] = {}

    with conn.cursor() as cur:
        for slot, query in {
            **CATEGORICAL_POOL_QUERIES,
            **CALENDAR_POOL_QUERIES,
        }.items():
            cur.execute(query)
            values = [row[0] for row in cur.fetchall() if row[0] is not None]
            if not values:
                raise RuntimeError(
                    f"Value pool {slot!r} came back empty. Has the database "
                    "been populated by scripts/generate_data.py?"
                )
            pools[slot] = values

        for slot, spec in NUMERIC_POOLS.items():
            query = _discrete_query(spec) if spec.discrete else _continuous_query(spec)
            cur.execute(query)
            values = [
                _coerce(row[0], spec.digits)
                for row in cur.fetchall()
                if row[0] is not None
            ]
            if not values:
                raise RuntimeError(
                    f"Numeric pool {slot!r} produced no interior values. The "
                    "underlying distribution may be constant, which would make "
                    "every threshold either vacuous or empty."
                )
            pools[slot] = values

    pools.update(STRUCTURAL_POOLS)
    return ValuePools(pools)


def sql_escape(value: Any) -> str:
    """Escape a value for inclusion in a SQL string literal.

    Single quotes are doubled, the SQL standard escape. Templates supply the
    surrounding quotes, so this handles only the inside.
    """
    return str(value).replace("'", "''")


def missing_pools(slots: Sequence[str], pools: ValuePools) -> list[str]:
    """Return slots with no pool, for validating templates up front."""
    return [s for s in slots if s not in pools]
