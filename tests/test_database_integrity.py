"""Database integrity tests.

Phase 1, Task 1.5.

These verify the database is trustworthy enough to build a benchmark on. If the
data is inconsistent, every downstream number — baseline accuracy, fine-tuned
accuracy, the whole ablation study — is measuring the wrong thing.

The tests query the database directly rather than re-using the generator's
logic. A test that shares the generator's assumptions would pass even when both
are wrong.

Run from the project root:

    env\\Scripts\\python.exe -m pytest tests/ -v
"""

from __future__ import annotations

from datetime import date

import psycopg
import pytest

from src.sql.config import app_config

# Minimum row counts required by Task 1.4.
REQUIRED_VOLUMES = {
    "customers": 10_000,
    "orders": 50_000,
    "order_items": 150_000,
    "products": 2_000,
    "categories": 100,
    "suppliers": 200,
    "payments": 50_000,
}

ALL_TABLES = [
    "categories", "suppliers", "warehouses", "departments", "employees",
    "customers", "products", "inventory", "orders", "order_items",
    "payments", "shipments",
]


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_all_tables_exist(query):
    found = {
        r[0] for r in query(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
        )
    }
    assert set(ALL_TABLES) <= found, f"missing tables: {set(ALL_TABLES) - found}"


@pytest.mark.parametrize("table,minimum", sorted(REQUIRED_VOLUMES.items()))
def test_row_count_meets_target(scalar, table: str, minimum: int):
    count = scalar(f"SELECT COUNT(*) FROM {table}")
    assert count >= minimum, f"{table} has {count:,}, need >= {minimum:,}"


@pytest.mark.parametrize("table", ALL_TABLES)
def test_table_is_not_empty(scalar, table: str):
    assert scalar(f"SELECT COUNT(*) FROM {table}") > 0


# ---------------------------------------------------------------------------
# Referential integrity
#
# Foreign keys should make orphans impossible. These tests confirm the
# constraints are actually enforced rather than merely declared — a table
# created without its constraint, or loaded with them disabled, looks fine
# until a join silently returns nothing.
# ---------------------------------------------------------------------------

ORPHAN_CHECKS = [
    ("orders", "customer_id", "customers", "customer_id"),
    ("orders", "employee_id", "employees", "employee_id"),
    ("order_items", "order_id", "orders", "order_id"),
    ("order_items", "product_id", "products", "product_id"),
    ("products", "category_id", "categories", "category_id"),
    ("products", "supplier_id", "suppliers", "supplier_id"),
    ("payments", "order_id", "orders", "order_id"),
    ("shipments", "order_id", "orders", "order_id"),
    ("shipments", "warehouse_id", "warehouses", "warehouse_id"),
    ("inventory", "product_id", "products", "product_id"),
    ("inventory", "warehouse_id", "warehouses", "warehouse_id"),
    ("employees", "department_id", "departments", "department_id"),
    ("employees", "manager_id", "employees", "employee_id"),
    ("categories", "parent_category_id", "categories", "category_id"),
]


@pytest.mark.parametrize("child,fk,parent,pk", ORPHAN_CHECKS)
def test_no_orphaned_rows(scalar, child, fk, parent, pk):
    """Every non-NULL foreign key points at a row that exists."""
    orphans = scalar(
        f"SELECT COUNT(*) FROM {child} c "
        f"LEFT JOIN {parent} p ON c.{fk} = p.{pk} "
        f"WHERE c.{fk} IS NOT NULL AND p.{pk} IS NULL"
    )
    assert orphans == 0, f"{orphans} orphaned {child}.{fk}"


def test_foreign_key_is_actually_enforced():
    """Inserting a deliberately invalid foreign key must be rejected.

    Uses its own writable connection and always rolls back, so the database is
    unchanged whether the test passes or fails.
    """
    with psycopg.connect(app_config().conninfo()) as conn:
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO orders (customer_id, order_date, total_amount) "
                    "VALUES (-999, NOW(), 10.00)"
                )
        conn.rollback()


def test_check_constraint_is_actually_enforced():
    """A negative quantity must be rejected by chk_order_items_quantity_positive."""
    with psycopg.connect(app_config().conninfo()) as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO order_items "
                    "(order_id, product_id, quantity, unit_price) "
                    "VALUES (1, 1, -5, 10.00)"
                )
        conn.rollback()


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------

NOT_NULL_COLUMNS = [
    ("customers", "customer_name"), ("customers", "email"),
    ("customers", "country"), ("customers", "signup_date"),
    ("orders", "customer_id"), ("orders", "order_date"),
    ("orders", "status"), ("orders", "total_amount"),
    ("order_items", "quantity"), ("order_items", "unit_price"),
    ("products", "product_name"), ("products", "sku"),
    ("products", "unit_price"), ("products", "category_id"),
    ("employees", "first_name"), ("employees", "salary"),
    ("payments", "amount"), ("payments", "payment_method"),
]


@pytest.mark.parametrize("table,column", NOT_NULL_COLUMNS)
def test_required_field_has_no_nulls(scalar, table: str, column: str):
    nulls = scalar(f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL")
    assert nulls == 0, f"{nulls} NULLs in {table}.{column}"


UNIQUE_COLUMNS = [
    ("customers", "email"), ("employees", "email"), ("products", "sku"),
    ("suppliers", "supplier_name"), ("categories", "category_name"),
    ("shipments", "tracking_number"), ("payments", "transaction_ref"),
]


@pytest.mark.parametrize("table,column", UNIQUE_COLUMNS)
def test_no_duplicate_values(scalar, table: str, column: str):
    dupes = scalar(
        f"SELECT COUNT(*) FROM (SELECT {column} FROM {table} "
        f"WHERE {column} IS NOT NULL GROUP BY {column} HAVING COUNT(*) > 1) d"
    )
    assert dupes == 0, f"{dupes} duplicate values in {table}.{column}"


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def test_no_orders_before_customer_signup(scalar):
    """A customer cannot order before they existed."""
    bad = scalar(
        "SELECT COUNT(*) FROM orders o JOIN customers c USING (customer_id) "
        "WHERE o.order_date::date < c.signup_date"
    )
    assert bad == 0, f"{bad} orders predate their customer's signup"


def test_no_future_order_dates(scalar):
    bad = scalar("SELECT COUNT(*) FROM orders WHERE order_date > NOW()")
    assert bad == 0, f"{bad} orders dated in the future"


def test_delivery_never_precedes_shipping(scalar):
    bad = scalar(
        "SELECT COUNT(*) FROM shipments "
        "WHERE delivered_date IS NOT NULL AND shipped_date IS NOT NULL "
        "AND delivered_date < shipped_date"
    )
    assert bad == 0, f"{bad} shipments delivered before they shipped"


def test_signup_dates_within_expected_range(scalar):
    bad = scalar(
        "SELECT COUNT(*) FROM customers "
        "WHERE signup_date < DATE '2015-01-01' OR signup_date > CURRENT_DATE"
    )
    assert bad == 0, f"{bad} customers have implausible signup dates"


def test_hire_dates_are_plausible(scalar):
    bad = scalar(
        "SELECT COUNT(*) FROM employees "
        "WHERE hire_date < DATE '1990-01-01' OR hire_date > CURRENT_DATE"
    )
    assert bad == 0, f"{bad} employees have implausible hire dates"


def test_payments_not_before_their_order(scalar):
    """Compared as timestamps, not dates.

    Casting to ::date converts to the session timezone first, so the same data
    can pass or fail depending on the client's timezone. The real invariant is
    on the instant: a payment cannot be recorded before its order exists.
    """
    bad = scalar(
        "SELECT COUNT(*) FROM payments p JOIN orders o USING (order_id) "
        "WHERE p.payment_date < o.order_date"
    )
    assert bad == 0, f"{bad} payments predate their order"


def test_shipments_not_before_their_order(scalar):
    bad = scalar(
        "SELECT COUNT(*) FROM shipments s JOIN orders o USING (order_id) "
        "WHERE s.shipped_date IS NOT NULL AND s.shipped_date < o.order_date"
    )
    assert bad == 0, f"{bad} shipments predate their order"


def test_no_future_payment_or_delivery_dates(scalar):
    future_payments = scalar("SELECT COUNT(*) FROM payments WHERE payment_date > NOW()")
    future_deliveries = scalar(
        "SELECT COUNT(*) FROM shipments WHERE delivered_date > NOW()"
    )
    assert future_payments == 0, f"{future_payments} payments dated in the future"
    assert future_deliveries == 0, f"{future_deliveries} deliveries in the future"


# ---------------------------------------------------------------------------
# Prices and amounts
# ---------------------------------------------------------------------------

def test_no_negative_prices(scalar):
    assert scalar("SELECT COUNT(*) FROM products WHERE unit_price < 0") == 0
    assert scalar("SELECT COUNT(*) FROM order_items WHERE unit_price < 0") == 0


def test_prices_are_within_a_plausible_range(scalar):
    """Guards against a generator bug producing absurd magnitudes."""
    lo, hi = scalar("SELECT MIN(unit_price) FROM products"), scalar(
        "SELECT MAX(unit_price) FROM products")
    assert 0 < lo < 100, f"minimum product price {lo} looks wrong"
    assert 100 < hi < 100_000, f"maximum product price {hi} looks wrong"


def test_payment_amounts_are_positive(scalar):
    assert scalar("SELECT COUNT(*) FROM payments WHERE amount <= 0") == 0


def test_discounts_within_range(scalar):
    bad = scalar(
        "SELECT COUNT(*) FROM order_items "
        "WHERE discount_pct < 0 OR discount_pct > 100"
    )
    assert bad == 0


def test_salaries_are_positive(scalar):
    assert scalar("SELECT COUNT(*) FROM employees WHERE salary <= 0") == 0


def test_inventory_quantities_non_negative(scalar):
    bad = scalar("SELECT COUNT(*) FROM inventory WHERE quantity_on_hand < 0")
    assert bad == 0


# ---------------------------------------------------------------------------
# Business invariants
# ---------------------------------------------------------------------------

def test_every_order_has_at_least_one_item(scalar):
    empty = scalar(
        "SELECT COUNT(*) FROM orders o "
        "WHERE NOT EXISTS (SELECT 1 FROM order_items i WHERE i.order_id = o.order_id)"
    )
    assert empty == 0, f"{empty} orders have no line items"


def test_order_total_matches_sum_of_its_lines(scalar):
    """orders.total_amount is denormalised, so it must agree with the lines.

    This is the single most important data test. Benchmark questions ask for
    revenue both ways — from the column and from the lines — and if the two
    disagree, two correct queries return different answers and execution-based
    scoring becomes unreliable.
    """
    mismatched = scalar(
        """
        SELECT COUNT(*) FROM (
            SELECT o.order_id
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.order_id
            GROUP BY o.order_id, o.total_amount, o.shipping_cost
            HAVING o.total_amount <> ROUND(
                SUM(ROUND(oi.quantity * oi.unit_price
                          * (1 - oi.discount_pct / 100), 2))
                + o.shipping_cost, 2)
        ) bad
        """
    )
    assert mismatched == 0, f"{mismatched} orders whose total != sum(lines) + shipping"


def test_cancelled_orders_have_no_payments(scalar):
    bad = scalar(
        "SELECT COUNT(*) FROM payments p JOIN orders o USING (order_id) "
        "WHERE o.status = 'cancelled'"
    )
    assert bad == 0, f"{bad} payments against cancelled orders"


def test_products_belong_to_valid_categories(scalar):
    """Restates the FK as an explicit business rule, per Task 1.5."""
    bad = scalar(
        "SELECT COUNT(*) FROM products p "
        "LEFT JOIN categories c ON p.category_id = c.category_id "
        "WHERE c.category_id IS NULL"
    )
    assert bad == 0, f"{bad} products in nonexistent categories"


def test_no_category_is_its_own_parent(scalar):
    bad = scalar(
        "SELECT COUNT(*) FROM categories WHERE parent_category_id = category_id"
    )
    assert bad == 0


def test_no_employee_manages_themselves(scalar):
    bad = scalar("SELECT COUNT(*) FROM employees WHERE manager_id = employee_id")
    assert bad == 0


def test_exactly_one_top_level_employee(scalar):
    """One CEO. More than one root means the org chart is broken."""
    roots = scalar("SELECT COUNT(*) FROM employees WHERE manager_id IS NULL")
    assert roots == 1, f"expected 1 employee with no manager, found {roots}"


def test_employee_hierarchy_has_no_cycles(scalar):
    """Walk the reporting chain; every employee must reach the root.

    A cycle (A reports to B, B reports to A) makes recursive org-chart queries
    loop forever, so it must be impossible by construction.
    """
    reachable = scalar(
        """
        WITH RECURSIVE chain AS (
            SELECT employee_id FROM employees WHERE manager_id IS NULL
            UNION ALL
            SELECT e.employee_id
            FROM employees e JOIN chain c ON e.manager_id = c.employee_id
        )
        SELECT COUNT(*) FROM chain
        """
    )
    total = scalar("SELECT COUNT(*) FROM employees")
    assert reachable == total, (
        f"only {reachable}/{total} employees reachable from the root — "
        "the hierarchy contains a cycle or an orphaned branch"
    )


def test_category_hierarchy_has_no_cycles(scalar):
    reachable = scalar(
        """
        WITH RECURSIVE tree AS (
            SELECT category_id FROM categories WHERE parent_category_id IS NULL
            UNION ALL
            SELECT c.category_id
            FROM categories c JOIN tree t ON c.parent_category_id = t.category_id
        )
        SELECT COUNT(*) FROM tree
        """
    )
    total = scalar("SELECT COUNT(*) FROM categories")
    assert reachable == total, f"only {reachable}/{total} categories reachable"


# ---------------------------------------------------------------------------
# Benchmark suitability
#
# The schema features that make hard questions possible must actually be
# present in the data. A nullable column that happens to contain no NULLs
# cannot test NULL handling.
# ---------------------------------------------------------------------------

NULLABLE_COLUMNS_THAT_MUST_HAVE_NULLS = [
    ("orders", "employee_id"),
    ("products", "supplier_id"),
    ("customers", "credit_limit"),
    ("shipments", "delivered_date"),
    ("suppliers", "rating"),
]


@pytest.mark.parametrize("table,column", NULLABLE_COLUMNS_THAT_MUST_HAVE_NULLS)
def test_nullable_column_actually_contains_nulls(scalar, table: str, column: str):
    nulls = scalar(f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL")
    assert nulls > 0, (
        f"{table}.{column} has no NULLs, so NULL-handling questions "
        "cannot be tested against it"
    )


def test_category_hierarchy_has_depth(scalar):
    """Sub-categories must exist, or recursive-CTE questions are trivial."""
    children = scalar(
        "SELECT COUNT(*) FROM categories WHERE parent_category_id IS NOT NULL"
    )
    assert children > 50, f"only {children} sub-categories"


def test_orders_span_multiple_years(scalar):
    """Date-comparison questions need a real time range to work with."""
    years = scalar("SELECT COUNT(DISTINCT EXTRACT(YEAR FROM order_date)) FROM orders")
    assert years >= 3, f"orders span only {years} year(s)"


def test_partially_paid_orders_exist(scalar):
    """Under-payment must occur, or HAVING SUM(...) questions have no answer."""
    partial = scalar(
        """
        SELECT COUNT(*) FROM (
            SELECT o.order_id
            FROM orders o
            JOIN payments p ON p.order_id = o.order_id AND p.status = 'completed'
            WHERE o.status <> 'cancelled'
            GROUP BY o.order_id, o.total_amount
            HAVING SUM(p.amount) < o.total_amount
        ) x
        """
    )
    assert partial > 0, "no partially paid orders exist"


def test_all_order_statuses_are_represented(scalar):
    found = scalar("SELECT COUNT(DISTINCT status) FROM orders")
    assert found == 6, f"expected all 6 order statuses in the data, found {found}"
