"""Generate realistic synthetic data for the enterprise_sql database.

Phase 1, Task 1.4.

No real personal information is used. Names are composed from generic word
pools and all email addresses use example.com, a domain reserved by RFC 2606
precisely so it can never belong to anyone.

Two properties matter more than realism:

*Reproducibility* — a fixed RANDOM_SEED means this script always produces the
identical database. The Phase 4 baseline and the Phase 10 fine-tuned evaluation
must run against the same rows, or the comparison between them proves nothing.

*Internal consistency* — orders.total_amount really equals the sum of its lines
plus shipping, delivery dates really follow shipping dates, and every order has
at least one line. Task 1.5 verifies all of this independently.

Loading uses COPY rather than INSERT. COPY streams rows in a single pass and is
roughly two orders of magnitude faster; 165k INSERT statements would take
minutes on this hardware.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/generate_data.py
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import DATA_AS_OF  # noqa: E402
from src.sql.config import ConfigError, app_config  # noqa: E402

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

RANDOM_SEED = 20260808

N_DEPARTMENTS = 12
N_EMPLOYEES = 500
N_TOP_CATEGORIES = 20
N_CATEGORIES = 120
N_SUPPLIERS = 250
N_WAREHOUSES = 15
N_PRODUCTS = 2_500
N_CUSTOMERS = 12_000
N_ORDERS = 55_000

# Calendar bounds. Kept in the past so "last month" style questions are stable.
EARLIEST_SIGNUP = date(2019, 1, 1)
LATEST_SIGNUP = date(2026, 5, 1)
EARLIEST_ORDER = date(2019, 1, 15)
# Orders stop a month before the dataset's as-of date. That gap leaves room
# for the longest downstream chain — ship up to 7 days after the order, then
# deliver up to 14 days later — to complete without running past DATA_AS_OF.
LATEST_ORDER = date(2026, 7, 1)

# The instant the dataset is notionally "as of". Payments, shipments and
# deliveries are clamped to it so nothing can be dated in the future. A fixed
# constant rather than the real clock, because the generator must be
# reproducible — using NOW() would make the data depend on the run date.
# With the gap above it acts as a safety net that should never actually bind;
# if it did, rows would bunch up at exactly this timestamp.
#
# Imported rather than defined here: the benchmark's time-relative questions
# anchor to the same instant, and two copies of the value would eventually
# disagree.  See src/constants.py.

UTC = timezone.utc
TWO_DP = Decimal("0.01")

# --------------------------------------------------------------------------
# Deliberately unreferenced entities
#
# The first version of this generator distributed foreign keys uniformly, so
# every product was ordered, every warehouse stocked, every supplier supplying.
# That is unrealistic — real enterprise databases are full of SKUs nobody buys,
# categories created ahead of a launch, and warehouses being commissioned.
#
# It also made a whole class of question unanswerable. "Which products have
# never been ordered?" correctly returned zero rows, and a gold query returning
# nothing cannot grade anything: a completely wrong query returns nothing too.
# Phase 3 rejected six templates for exactly this reason.
#
# These fractions restore the gaps distributionally. Nothing here targets a
# specific benchmark answer — which entities end up unreferenced is decided by
# the seeded RNG, and the counts move if the seed or the volumes change.
# --------------------------------------------------------------------------

FRACTION_CATEGORIES_WITHOUT_PRODUCTS = 0.08     # created ahead of a launch
FRACTION_SUPPLIERS_WITHOUT_PRODUCTS = 0.06      # onboarded, not yet supplying
FRACTION_PRODUCTS_NEVER_ORDERED = 0.06          # new SKUs and dead stock
FRACTION_PRODUCTS_AT_OR_BELOW_COST = 0.025      # loss leaders and clearance
N_WAREHOUSES_WITHOUT_INVENTORY = 2              # cross-dock / commissioning
FRACTION_EMPLOYEES_HANDLING_ORDERS = 0.55       # only customer-facing staff sell

# --------------------------------------------------------------------------
# Word pools — generic, synthetic, not derived from any real dataset
# --------------------------------------------------------------------------

FIRST_NAMES = [
    "Aarav", "Amara", "Bilal", "Camila", "Daniel", "Elena", "Farah", "Gabriel",
    "Hana", "Ibrahim", "Julia", "Kenji", "Lucia", "Mateo", "Nadia", "Omar",
    "Priya", "Quinn", "Rohan", "Sofia", "Tomas", "Uma", "Viktor", "Wen",
    "Ximena", "Yusuf", "Zara", "Anders", "Beatriz", "Chen", "Diego", "Freya",
    "Giulia", "Hassan", "Ingrid", "Jamal", "Kavya", "Lars", "Mei", "Niamh",
]

LAST_NAMES = [
    "Adeyemi", "Baptiste", "Chowdhury", "Delacroix", "Escobar", "Fitzgerald",
    "Gunnarsson", "Haddad", "Iyer", "Jansen", "Kowalski", "Lindqvist",
    "Mbeki", "Nakamura", "Okafor", "Petrov", "Quintero", "Rasmussen",
    "Silva", "Tanaka", "Ueda", "Vargas", "Wagner", "Xu", "Yamamoto", "Zielinski",
    "Almeida", "Bergström", "Castellanos", "Dubois", "Eriksen", "Ferrari",
]

COMPANY_HEADS = [
    "Northwind", "Vertex", "Lumina", "Ironclad", "Summit", "Cobalt", "Aurora",
    "Meridian", "Pinnacle", "Cascade", "Beacon", "Quarry", "Zenith", "Harbor",
    "Foundry", "Thistle", "Orchid", "Granite", "Tempest", "Lantern", "Compass",
    "Bastion", "Juniper", "Kestrel", "Onyx", "Palisade", "Redwood", "Sable",
]

COMPANY_TAILS = [
    "Industries", "Trading", "Supply Co", "Holdings", "Partners", "Group",
    "Logistics", "Systems", "Works", "Distribution", "Enterprises", "Global",
]

COUNTRIES = [
    ("India", ["Mumbai", "Bengaluru", "Delhi", "Chennai", "Hyderabad", "Pune"]),
    ("United States", ["Austin", "Seattle", "Chicago", "Denver", "Boston"]),
    ("Germany", ["Berlin", "Munich", "Hamburg", "Frankfurt"]),
    ("Japan", ["Tokyo", "Osaka", "Nagoya", "Fukuoka"]),
    ("Brazil", ["Sao Paulo", "Rio de Janeiro", "Belo Horizonte"]),
    ("United Kingdom", ["London", "Manchester", "Bristol", "Leeds"]),
    ("Canada", ["Toronto", "Vancouver", "Montreal"]),
    ("Australia", ["Sydney", "Melbourne", "Brisbane"]),
    ("France", ["Paris", "Lyon", "Marseille"]),
    ("Singapore", ["Singapore"]),
    ("Netherlands", ["Amsterdam", "Rotterdam"]),
    ("South Africa", ["Cape Town", "Johannesburg"]),
]

TOP_CATEGORIES = [
    "Electronics", "Home & Kitchen", "Office Supplies", "Industrial Tools",
    "Apparel", "Sports & Outdoors", "Automotive", "Health & Beauty",
    "Toys & Games", "Garden", "Pet Supplies", "Books & Media",
    "Food & Beverage", "Furniture", "Lighting", "Networking",
    "Safety Equipment", "Packaging", "Cleaning Supplies", "Laboratory",
]

SUBCATEGORY_WORDS = [
    "Accessories", "Components", "Premium", "Essentials", "Professional",
    "Compact", "Heavy Duty", "Portable", "Wireless", "Replacement Parts",
    "Starter Kits", "Bulk", "Refurbished", "Commercial", "Consumer",
]

PRODUCT_ADJECTIVES = [
    "Compact", "Industrial", "Premium", "Ergonomic", "Rugged", "Wireless",
    "Adjustable", "Insulated", "Stainless", "Modular", "Precision", "Heavy-Duty",
    "Lightweight", "Waterproof", "Reinforced", "Digital", "Automatic",
]

PRODUCT_NOUNS = [
    "Router", "Desk Lamp", "Toolkit", "Backpack", "Monitor Stand", "Keyboard",
    "Water Bottle", "Filter Cartridge", "Cable Set", "Storage Bin", "Headset",
    "Power Adapter", "Label Printer", "Safety Vest", "Torque Wrench",
    "Air Purifier", "Desk Organiser", "Barcode Scanner", "Floor Mat",
    "Pressure Gauge", "Hand Truck", "Shelf Unit", "Drill Bit Set", "Extension Reel",
]

DEPARTMENT_NAMES = [
    "Sales", "Marketing", "Engineering", "Customer Support", "Finance",
    "Human Resources", "Operations", "Procurement", "Logistics",
    "Quality Assurance", "Legal", "IT",
]

JOB_TITLES_BY_LEVEL = {
    0: ["Chief Executive Officer"],
    1: ["VP of Sales", "VP of Operations", "VP of Engineering", "VP of Finance"],
    2: ["Senior Manager", "Regional Manager", "Team Lead", "Programme Manager"],
    3: ["Analyst", "Specialist", "Coordinator", "Associate", "Representative"],
}

CARRIERS = ["GlobalEx", "SwiftFreight", "BluePort", "AeroCargo", "RegionalPost"]
PAYMENT_METHODS = [
    "credit_card", "credit_card", "credit_card", "debit_card",
    "paypal", "bank_transfer", "wire", "cash",
]
ORDER_STATUS_WEIGHTS = [
    ("delivered", 58), ("shipped", 12), ("processing", 10),
    ("pending", 8), ("cancelled", 8), ("returned", 4),
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def money(value: Decimal | float) -> Decimal:
    """Round to 2 decimal places the way currency is rounded.

    Python's default rounding is banker's rounding (0.5 goes to even), which
    does not match how money is normally rounded and would make computed
    totals disagree with hand-checked expectations.
    """
    return Decimal(value).quantize(TWO_DP, rounding=ROUND_HALF_UP)


def random_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def as_utc(day: date, rng: random.Random) -> datetime:
    """Attach a plausible business-hours time and UTC timezone to a date."""
    return datetime(
        day.year, day.month, day.day,
        rng.randint(7, 20), rng.randint(0, 59), rng.randint(0, 59),
        tzinfo=UTC,
    )


def weighted_choice(rng: random.Random, weighted: Sequence[tuple[str, int]]) -> str:
    total = sum(w for _, w in weighted)
    roll = rng.uniform(0, total)
    upto = 0.0
    for value, weight in weighted:
        upto += weight
        if roll <= upto:
            return value
    return weighted[-1][0]


def copy_rows(
    conn: psycopg.Connection,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> int:
    """Bulk-load rows with COPY. Returns the number of rows written."""
    count = 0
    cols = ", ".join(columns)
    with conn.cursor() as cur:
        with cur.copy(f"COPY {table} ({cols}) FROM STDIN") as copy:
            for row in rows:
                copy.write_row(row)
                count += 1
    return count


def reset_sequence(conn: psycopg.Connection, table: str, column: str) -> None:
    """Realign a SERIAL sequence after explicit IDs were supplied by COPY.

    COPY with explicit id values does not advance the sequence, so the next
    ordinary INSERT would collide with an existing row. This fixes that.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT setval(pg_get_serial_sequence('{table}', '{column}'), "
            f"COALESCE((SELECT MAX({column}) FROM {table}), 1))"
        )


# --------------------------------------------------------------------------
# Generators — each returns the rows it inserted so later tables can
# reference real IDs rather than guessing at them.
# --------------------------------------------------------------------------

def gen_departments(rng: random.Random) -> list[tuple]:
    return [
        (i, name, rng.choice(COUNTRIES)[1][0], money(rng.uniform(250_000, 5_000_000)))
        for i, name in enumerate(DEPARTMENT_NAMES[:N_DEPARTMENTS], start=1)
    ]


@dataclass(frozen=True)
class Reservations:
    """Which entities are deliberately left unreferenced.

    Decided once, up front, from a dedicated RNG stream so that changing these
    fractions does not shift every other random draw in the generator.
    """

    categories_with_products: tuple[int, ...]
    suppliers_with_products: tuple[int, ...]
    sellable_products: tuple[int, ...]
    products_at_or_below_cost: frozenset[int]
    stocked_warehouses: tuple[int, ...]
    order_handling_employees: tuple[int, ...]

    def describe(self) -> str:
        return (
            f"  categories with no products : "
            f"{N_CATEGORIES - len(self.categories_with_products)}\n"
            f"  suppliers with no products  : "
            f"{N_SUPPLIERS - len(self.suppliers_with_products)}\n"
            f"  products never ordered      : "
            f"{N_PRODUCTS - len(self.sellable_products)}\n"
            f"  products at or below cost   : "
            f"{len(self.products_at_or_below_cost)}\n"
            f"  warehouses with no stock    : "
            f"{N_WAREHOUSES - len(self.stocked_warehouses)}\n"
            f"  employees never handling an order: "
            f"{N_EMPLOYEES - len(self.order_handling_employees)}"
        )


def _keep_fraction(
    rng: random.Random, population: int, excluded_fraction: float
) -> tuple[int, ...]:
    """Return a sorted subset of 1..population, dropping the given fraction."""
    keep = population - round(population * excluded_fraction)
    return tuple(sorted(rng.sample(range(1, population + 1), keep)))


def plan_reservations(seed: int) -> Reservations:
    rng = random.Random(seed + 1)
    sellable = _keep_fraction(rng, N_PRODUCTS, FRACTION_PRODUCTS_NEVER_ORDERED)
    loss_leaders = rng.sample(
        range(1, N_PRODUCTS + 1),
        round(N_PRODUCTS * FRACTION_PRODUCTS_AT_OR_BELOW_COST),
    )
    return Reservations(
        categories_with_products=_keep_fraction(
            rng, N_CATEGORIES, FRACTION_CATEGORIES_WITHOUT_PRODUCTS),
        suppliers_with_products=_keep_fraction(
            rng, N_SUPPLIERS, FRACTION_SUPPLIERS_WITHOUT_PRODUCTS),
        sellable_products=sellable,
        products_at_or_below_cost=frozenset(loss_leaders),
        stocked_warehouses=tuple(sorted(rng.sample(
            range(1, N_WAREHOUSES + 1),
            N_WAREHOUSES - N_WAREHOUSES_WITHOUT_INVENTORY))),
        order_handling_employees=tuple(sorted(rng.sample(
            range(1, N_EMPLOYEES + 1),
            round(N_EMPLOYEES * FRACTION_EMPLOYEES_HANDLING_ORDERS)))),
    )


def gen_categories(rng: random.Random) -> list[tuple]:
    """Top-level categories first, then children pointing at them."""
    rows: list[tuple] = []
    for i, name in enumerate(TOP_CATEGORIES[:N_TOP_CATEGORIES], start=1):
        rows.append((i, name, None, f"Top-level category: {name}"))

    next_id = N_TOP_CATEGORIES + 1
    while next_id <= N_CATEGORIES:
        parent_id = rng.randint(1, N_TOP_CATEGORIES)
        parent_name = TOP_CATEGORIES[parent_id - 1]
        word = SUBCATEGORY_WORDS[(next_id * 7) % len(SUBCATEGORY_WORDS)]
        name = f"{parent_name} - {word}"
        # category_name is UNIQUE; disambiguate collisions with the id.
        if any(r[1] == name for r in rows):
            name = f"{name} {next_id}"
        rows.append((next_id, name, parent_id, f"Subcategory of {parent_name}"))
        next_id += 1
    return rows


def gen_suppliers(rng: random.Random) -> list[tuple]:
    rows = []
    for i in range(1, N_SUPPLIERS + 1):
        country, cities = rng.choice(COUNTRIES)
        name = f"{rng.choice(COMPANY_HEADS)} {rng.choice(COMPANY_TAILS)} {i}"
        contact = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        rows.append((
            i, name, contact, f"supplier{i}@example.com",
            f"+1-555-{i:04d}", country, rng.choice(cities),
            # 15% unrated, so rating is genuinely nullable in the data.
            None if rng.random() < 0.15 else money(rng.uniform(1.5, 5.0)),
            rng.random() > 0.08,
        ))
    return rows


def gen_warehouses(rng: random.Random) -> list[tuple]:
    rows = []
    for i in range(1, N_WAREHOUSES + 1):
        country, cities = rng.choice(COUNTRIES)
        city = rng.choice(cities)
        rows.append((
            i, f"{city} Distribution Centre {i}", country, city,
            rng.randrange(20_000, 250_000, 1_000), rng.random() > 0.07,
        ))
    return rows


def gen_employees(rng: random.Random) -> list[tuple]:
    """Build a four-level reporting hierarchy: CEO -> VP -> manager -> staff."""
    rows: list[tuple] = []
    levels: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}

    def add(emp_id: int, level: int, manager_id: int | None) -> None:
        first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        hire = random_date(rng, date(2010, 1, 1), date(2026, 3, 1))
        base = {0: 320_000, 1: 190_000, 2: 120_000, 3: 62_000}[level]
        rows.append((
            emp_id, first, last,
            f"{first.lower()}.{last.lower()}{emp_id}@example.com",
            f"+1-555-1{emp_id:04d}", hire,
            rng.choice(JOB_TITLES_BY_LEVEL[level]),
            money(base * rng.uniform(0.85, 1.3)),
            rng.randint(1, N_DEPARTMENTS), manager_id,
            rng.random() > 0.06,
        ))
        levels[level].append(emp_id)

    add(1, 0, None)                                   # CEO, manager_id NULL
    next_id = 2
    for _ in range(6):                                # VPs report to the CEO
        add(next_id, 1, 1)
        next_id += 1
    for _ in range(40):                               # managers report to VPs
        add(next_id, 2, rng.choice(levels[1]))
        next_id += 1
    while next_id <= N_EMPLOYEES:                     # staff report to managers
        add(next_id, 3, rng.choice(levels[2]))
        next_id += 1
    return rows


def gen_customers(rng: random.Random) -> list[tuple]:
    rows = []
    for i in range(1, N_CUSTOMERS + 1):
        country, cities = rng.choice(COUNTRIES)
        business = rng.random() < 0.3
        if business:
            name = f"{rng.choice(COMPANY_HEADS)} {rng.choice(COMPANY_TAILS)}"
            segment = "business"
            # Only business customers get a credit limit; individuals stay NULL.
            credit = money(rng.randrange(5_000, 500_000, 1_000))
        else:
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            segment = "individual"
            credit = None
        rows.append((
            i, name, f"customer{i}@example.com", f"+1-555-2{i:05d}",
            country, rng.choice(cities), segment, credit,
            random_date(rng, EARLIEST_SIGNUP, LATEST_SIGNUP),
            rng.random() > 0.05,
        ))
    return rows


def gen_products(rng: random.Random, reserved: Reservations) -> list[tuple]:
    rows = []
    for i in range(1, N_PRODUCTS + 1):
        name = (f"{rng.choice(PRODUCT_ADJECTIVES)} {rng.choice(PRODUCT_NOUNS)} "
                f"{rng.choice(['Mk I', 'Mk II', 'Pro', 'Lite', 'X', '2000'])}")
        price = money(rng.choice([
            rng.uniform(5, 60), rng.uniform(60, 400), rng.uniform(400, 3_000),
        ]))
        if i in reserved.products_at_or_below_cost:
            # Loss leaders and clearance lines: cost meets or exceeds the price.
            # Margin goes negative, which is what makes margin questions worth
            # asking rather than uniformly positive.
            cost = money(price * Decimal(str(rng.uniform(1.00, 1.25))))
        else:
            # Cost is 45-80% of list price, a realistic margin spread.
            cost = money(price * Decimal(str(rng.uniform(0.45, 0.80))))
        rows.append((
            i, f"{name} {i}", f"SKU-{i:06d}",
            # Drawn from the categories allowed to hold products, so the rest
            # stay genuinely empty.
            rng.choice(reserved.categories_with_products),
            # 12% have no supplier — a genuinely nullable foreign key. The rest
            # come from suppliers allowed to supply.
            None if rng.random() < 0.12 else rng.choice(reserved.suppliers_with_products),
            price, cost, money(rng.uniform(0.05, 40.0)),
            rng.random() < 0.10,
        ))
    return rows


def gen_inventory(rng: random.Random, reserved: Reservations) -> list[tuple]:
    """Each product is stocked in 1-4 warehouses, one row per pair.

    Warehouses outside ``stocked_warehouses`` receive nothing: they still
    dispatch shipments, as a cross-dock facility would, but hold no stock.
    """
    rows = []
    inv_id = 1
    for product_id in range(1, N_PRODUCTS + 1):
        for wh in rng.sample(reserved.stocked_warehouses, rng.randint(1, 4)):
            qty = rng.randint(0, 900)
            rows.append((
                inv_id, product_id, wh, qty, rng.randint(10, 120),
                as_utc(random_date(rng, date(2026, 1, 1), date(2026, 8, 1)), rng),
            ))
            inv_id += 1
    return rows


def gen_sales(
    rng: random.Random,
    customers: list[tuple],
    products: list[tuple],
    reserved: Reservations,
) -> tuple[list, list, list, list]:
    """Generate orders and everything hanging off them, kept consistent.

    Returns (orders, order_items, payments, shipments).
    """
    signup_by_customer = {row[0]: row[8] for row in customers}
    price_by_product = {row[0]: row[5] for row in products}

    orders, items, payments, shipments = [], [], [], []
    item_id = payment_id = shipment_id = 1

    for order_id in range(1, N_ORDERS + 1):
        customer_id = rng.randint(1, N_CUSTOMERS)
        # An order can never predate the customer's signup.
        signup = signup_by_customer[customer_id]
        earliest = max(signup, EARLIEST_ORDER)
        if earliest >= LATEST_ORDER:
            earliest = LATEST_ORDER - timedelta(days=1)
        order_day = random_date(rng, earliest, LATEST_ORDER)
        order_ts = as_utc(order_day, rng)
        status = weighted_choice(rng, ORDER_STATUS_WEIGHTS)

        # --- order lines ---
        n_lines = rng.choices([1, 2, 3, 4, 5, 6], weights=[22, 26, 22, 15, 9, 6])[0]
        # Only sellable products can appear on an order; the rest are new SKUs
        # and dead stock that exist in the catalogue but have never sold.
        chosen = rng.sample(reserved.sellable_products, n_lines)
        line_total = Decimal("0.00")
        for product_id in chosen:
            qty = rng.choices([1, 2, 3, 5, 10], weights=[45, 25, 15, 10, 5])[0]
            # Sale price drifts +-15% around list, so historical prices differ
            # from products.unit_price - as they would in a real system.
            unit = money(price_by_product[product_id] * Decimal(str(rng.uniform(0.85, 1.15))))
            discount = Decimal("0.00") if rng.random() < 0.7 else money(rng.choice([5, 10, 15, 20, 25]))
            line_total += money(qty * unit * (1 - discount / 100))
            items.append((item_id, order_id, product_id, qty, unit, discount))
            item_id += 1

        shipping = money(rng.uniform(0, 45))
        total = money(line_total + shipping)

        orders.append((
            order_id, customer_id,
            # 45% of orders are self-service online: employee_id stays NULL.
            # The rest go to customer-facing staff only — engineering, legal
            # and HR never handle a sale.
            None if rng.random() < 0.45
            else rng.choice(reserved.order_handling_employees),
            order_ts, order_day + timedelta(days=rng.randint(3, 21)),
            status, total, shipping, "USD",
        ))

        # --- payments: cancelled orders are never paid ---
        if status != "cancelled" and rng.random() > 0.06:
            if rng.random() < 0.2:
                # Deposit then balance: two rows summing to the total.
                deposit = money(total * Decimal("0.4"))
                schedule = [deposit, money(total - deposit)]
            else:
                schedule = [total]
            for n, amount in enumerate(schedule):
                if amount <= 0:
                    continue
                # Offset from the order's own timestamp, never an independently
                # drawn time — otherwise a same-day payment can land earlier in
                # the day than the order it pays for.
                pay_ts = order_ts + timedelta(
                    days=rng.randint(0, 5) + n * 14,
                    hours=rng.randint(1, 20),
                    minutes=rng.randint(0, 59),
                )
                payments.append((
                    payment_id, order_id, min(pay_ts, DATA_AS_OF),
                    amount, rng.choice(PAYMENT_METHODS),
                    "completed" if rng.random() > 0.04 else rng.choice(["failed", "refunded"]),
                    f"TXN-{payment_id:08d}",
                ))
                payment_id += 1

        # --- shipments: only for orders that actually left the warehouse ---
        if status in ("shipped", "delivered", "returned"):
            # Derived from order_ts, so a shipment can never predate its order.
            shipped_ts = min(
                order_ts + timedelta(days=rng.randint(0, 7), hours=rng.randint(1, 20)),
                DATA_AS_OF,
            )
            if status in ("delivered", "returned"):
                delivered_ts = min(
                    shipped_ts + timedelta(
                        days=rng.randint(1, 14), hours=rng.randint(1, 20)
                    ),
                    DATA_AS_OF,
                )
                ship_status = status
            else:
                # Shipped but not yet delivered: delivered_date stays NULL.
                delivered_ts = None
                ship_status = rng.choice(["in_transit", "in_transit", "delayed"])
            shipments.append((
                shipment_id, order_id, rng.randint(1, N_WAREHOUSES),
                shipped_ts, delivered_ts, rng.choice(CARRIERS),
                f"TRK-{shipment_id:09d}", ship_status,
            ))
            shipment_id += 1

    return orders, items, payments, shipments


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

TABLES_IN_LOAD_ORDER = [
    "categories", "suppliers", "warehouses", "departments", "employees",
    "customers", "products", "inventory", "orders", "order_items",
    "payments", "shipments",
]


def main() -> int:
    try:
        cfg = app_config()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    rng = random.Random(RANDOM_SEED)
    print(f"Seed: {RANDOM_SEED} (fixed - this database is reproducible)\n")

    reserved = plan_reservations(RANDOM_SEED)
    print("Deliberately unreferenced entities:")
    print(reserved.describe())

    print("\nGenerating...")
    departments = gen_departments(rng)
    categories = gen_categories(rng)
    suppliers = gen_suppliers(rng)
    warehouses = gen_warehouses(rng)
    employees = gen_employees(rng)
    customers = gen_customers(rng)
    products = gen_products(rng, reserved)
    inventory = gen_inventory(rng, reserved)
    orders, order_items, payments, shipments = gen_sales(
        rng, customers, products, reserved
    )
    print("  done\n")

    plan: list[tuple[str, Sequence[str], list]] = [
        ("categories",
         ["category_id", "category_name", "parent_category_id", "description"],
         categories),
        ("suppliers",
         ["supplier_id", "supplier_name", "contact_name", "contact_email",
          "phone", "country", "city", "rating", "is_active"], suppliers),
        ("warehouses",
         ["warehouse_id", "warehouse_name", "country", "city",
          "capacity_units", "is_active"], warehouses),
        ("departments",
         ["department_id", "department_name", "location", "annual_budget"],
         departments),
        ("employees",
         ["employee_id", "first_name", "last_name", "email", "phone",
          "hire_date", "job_title", "salary", "department_id", "manager_id",
          "is_active"], employees),
        ("customers",
         ["customer_id", "customer_name", "email", "phone", "country", "city",
          "customer_segment", "credit_limit", "signup_date", "is_active"],
         customers),
        ("products",
         ["product_id", "product_name", "sku", "category_id", "supplier_id",
          "unit_price", "unit_cost", "weight_kg", "is_discontinued"], products),
        ("inventory",
         ["inventory_id", "product_id", "warehouse_id", "quantity_on_hand",
          "reorder_level", "last_restocked_at"], inventory),
        ("orders",
         ["order_id", "customer_id", "employee_id", "order_date",
          "required_date", "status", "total_amount", "shipping_cost",
          "currency"], orders),
        ("order_items",
         ["order_item_id", "order_id", "product_id", "quantity", "unit_price",
          "discount_pct"], order_items),
        ("payments",
         ["payment_id", "order_id", "payment_date", "amount", "payment_method",
          "status", "transaction_ref"], payments),
        ("shipments",
         ["shipment_id", "order_id", "warehouse_id", "shipped_date",
          "delivered_date", "carrier", "tracking_number", "status"], shipments),
    ]

    try:
        with psycopg.connect(cfg.conninfo(), connect_timeout=10) as conn:
            # One transaction for the whole load: either the database ends up
            # fully populated and consistent, or entirely unchanged.
            with conn.cursor() as cur:
                cur.execute(
                    "TRUNCATE " + ", ".join(reversed(TABLES_IN_LOAD_ORDER))
                    + " RESTART IDENTITY CASCADE"
                )
            print("Existing rows cleared.\n")

            print(f"{'table':<15}{'rows':>10}")
            print("-" * 25)
            for table, columns, rows in plan:
                written = copy_rows(conn, table, columns, rows)
                print(f"{table:<15}{written:>10,}")

            for table, columns, _ in plan:
                reset_sequence(conn, table, columns[0])

            conn.commit()

            with conn.cursor() as cur:
                cur.execute("ANALYZE")

    except psycopg.Error as exc:
        print(f"\n[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\n[ok] data generated and committed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
