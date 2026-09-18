"""Question/SQL templates for the Text-to-SQL benchmark.

Phase 2.

Each template pairs several natural-language phrasings with one SQL query.
Slots written as {slot} appear in both and are filled with the same value, so
question and SQL always agree.

Why several phrasings per template: a model trained on one way of asking
"how many customers are in India" learns that exact sentence, not the task.
Paraphrases force it to generalise. They are near-duplicates of each other,
though, so all phrasings of a template must land in the same train/test split —
see scripts/generate_benchmark.py.

Difficulty tiers follow CLAUDE.md:

    easy        SELECT, WHERE, ORDER BY, LIMIT
    medium      GROUP BY, HAVING, COUNT, SUM, AVG
    hard        JOIN, multiple JOINs, subqueries, CTEs
    very_hard   window functions, nested aggregation, date comparison,
                conditional aggregation
    enterprise  ambiguous terminology, NULL handling, business rules,
                cross-domain joins
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.constants import DATA_AS_OF_SQL  # noqa: E402

SLOT_PATTERN = re.compile(r"\{(\w+)\}")

# Placeholder for the dataset's fixed "now". Templates write {as_of} and it is
# replaced with a literal timestamp when the template is constructed — before
# slot filling, so it never becomes a fillable slot needing a value pool, and
# never has to appear in the question text.
#
# Time-relative questions must not use NOW(). "Orders pending more than 90
# days" would return a different set every day, so the Phase 5 baseline and the
# Phase 10 fine-tuned run would be graded against different correct answers
# while still producing plausible-looking numbers.
AS_OF_PLACEHOLDER = "{as_of}"


@dataclass(frozen=True)
class Template:
    template_id: str
    difficulty: str
    domain: str
    query_type: tuple[str, ...]
    questions: tuple[str, ...]
    sql: str

    @property
    def slots(self) -> tuple[str, ...]:
        """Slot names used anywhere in the template, deduplicated, in order."""
        found: list[str] = []
        for text in (*self.questions, self.sql):
            for slot in SLOT_PATTERN.findall(text):
                if slot not in found:
                    found.append(slot)
        return tuple(found)


def T(
    template_id: str,
    difficulty: str,
    domain: str,
    query_type: str,
    questions: list[str],
    sql: str,
) -> Template:
    """Compact constructor. query_type is a space-separated feature list.

    {as_of} is resolved here rather than during slot filling, so the fixed
    timestamp is baked into the stored SQL. Anything that later reads a
    template — generation, validation, evaluation — sees the same anchored
    query without needing to know the constant exists.
    """
    return Template(
        template_id=template_id,
        difficulty=difficulty,
        domain=domain,
        query_type=tuple(query_type.split()),
        questions=tuple(questions),
        sql=" ".join(sql.replace(AS_OF_PLACEHOLDER, DATA_AS_OF_SQL).split()),
    )


# ===========================================================================
# EASY — SELECT, WHERE, ORDER BY, LIMIT
# ===========================================================================

EASY: list[Template] = [
    T("e01", "easy", "sales", "select where",
      ["Show all customers from {country}.",
       "List the customers located in {country}.",
       "Which customers are based in {country}?"],
      """SELECT customer_id, customer_name, country FROM customers
         WHERE country = '{country}'"""),

    T("e02", "easy", "sales", "select where",
      ["Show customers in the city of {city}.",
       "List all customers from {city}.",
       "Who are our customers in {city}?"],
      """SELECT customer_id, customer_name, city FROM customers
         WHERE city = '{city}'"""),

    T("e03", "easy", "sales", "select where",
      ["List all {segment} customers.",
       "Show every customer in the {segment} segment.",
       "Which customers are classified as {segment}?"],
      """SELECT customer_id, customer_name, customer_segment FROM customers
         WHERE customer_segment = '{segment}'"""),

    T("e04", "easy", "sales", "select where boolean",
      ["Show all inactive customers.",
       "Which customers are no longer active?",
       "List customers whose account is inactive."],
      """SELECT customer_id, customer_name FROM customers
         WHERE is_active = FALSE"""),

    T("e05", "easy", "sales", "select where order_by limit",
      ["Show the {n} most recent customer signups.",
       "List the newest {n} customers by signup date.",
       "Who are the last {n} customers to join?"],
      """SELECT customer_id, customer_name, signup_date FROM customers
         ORDER BY signup_date DESC LIMIT {n}"""),

    T("e06", "easy", "catalogue", "select where",
      ["Show products that cost more than {product_price} dollars.",
       "List all products priced above {product_price}.",
       "Which products have a unit price over {product_price}?"],
      """SELECT product_id, product_name, unit_price FROM products
         WHERE unit_price > {product_price}"""),

    T("e07", "easy", "catalogue", "select where",
      ["Show products cheaper than {product_price} dollars.",
       "List products with a price below {product_price}.",
       "Which products cost less than {product_price}?"],
      """SELECT product_id, product_name, unit_price FROM products
         WHERE unit_price < {product_price}"""),

    T("e08", "easy", "catalogue", "select where boolean",
      ["List all discontinued products.",
       "Which products have been discontinued?",
       "Show products that are no longer sold."],
      """SELECT product_id, product_name FROM products
         WHERE is_discontinued = TRUE"""),

    T("e09", "easy", "catalogue", "select order_by limit",
      ["What are the {n} most expensive products?",
       "Show the top {n} products by price.",
       "List our {n} priciest items."],
      """SELECT product_id, product_name, unit_price FROM products
         ORDER BY unit_price DESC LIMIT {n}"""),

    T("e10", "easy", "catalogue", "select order_by limit",
      ["What are the {n} cheapest products?",
       "Show the {n} lowest priced products.",
       "List the {n} least expensive items."],
      """SELECT product_id, product_name, unit_price FROM products
         ORDER BY unit_price ASC LIMIT {n}"""),

    T("e11", "easy", "catalogue", "select where",
      ["Show products heavier than {product_weight} kg.",
       "Which products weigh more than {product_weight} kilograms?",
       "List items with a weight above {product_weight} kg."],
      """SELECT product_id, product_name, weight_kg FROM products
         WHERE weight_kg > {product_weight}"""),

    T("e12", "easy", "sales", "select where",
      ["Show all orders with status {status}.",
       "List the orders that are currently {status}.",
       "Which orders have a {status} status?"],
      """SELECT order_id, customer_id, order_date, status FROM orders
         WHERE status = '{status}'"""),

    T("e13", "easy", "sales", "select where date",
      ["Show orders placed in {year}.",
       "List all orders from the year {year}.",
       "Which orders were made during {year}?"],
      """SELECT order_id, customer_id, order_date, total_amount FROM orders
         WHERE EXTRACT(YEAR FROM order_date) = {year}"""),

    T("e14", "easy", "sales", "select where",
      ["Show orders worth more than {order_total} dollars.",
       "List orders with a total above {order_total}.",
       "Which orders exceed {order_total} in value?"],
      """SELECT order_id, customer_id, total_amount FROM orders
         WHERE total_amount > {order_total}"""),

    T("e15", "easy", "sales", "select order_by limit",
      ["What are the {n} largest orders by value?",
       "Show the top {n} orders by total amount.",
       "List the {n} biggest orders we have received."],
      """SELECT order_id, customer_id, total_amount FROM orders
         ORDER BY total_amount DESC LIMIT {n}"""),

    T("e16", "easy", "hr", "select where",
      ["Show all employees with the job title {job_title}.",
       "List employees working as {job_title}.",
       "Who holds the position of {job_title}?"],
      """SELECT employee_id, first_name, last_name, job_title FROM employees
         WHERE job_title = '{job_title}'"""),

    T("e17", "easy", "hr", "select where order_by limit",
      ["Show the {n} highest paid employees.",
       "Who are the top {n} earners in the company?",
       "List the {n} employees with the largest salaries."],
      """SELECT employee_id, first_name, last_name, salary FROM employees
         ORDER BY salary DESC LIMIT {n}"""),

    T("e18", "easy", "hr", "select where",
      ["Show employees earning more than {employee_salary}.",
       "Which employees have a salary above {employee_salary}?",
       "List staff paid over {employee_salary}."],
      """SELECT employee_id, first_name, last_name, salary FROM employees
         WHERE salary > {employee_salary}"""),

    T("e19", "easy", "catalogue", "select where",
      ["Show suppliers based in {supplier_country}.",
       "List all suppliers from {supplier_country}.",
       "Which suppliers operate in {supplier_country}?"],
      """SELECT supplier_id, supplier_name, country FROM suppliers
         WHERE country = '{supplier_country}'"""),

    T("e20", "easy", "logistics", "select where boolean",
      ["List all active warehouses.",
       "Show the warehouses that are currently in use.",
       "Which warehouses are active?"],
      """SELECT warehouse_id, warehouse_name, city FROM warehouses
         WHERE is_active = TRUE"""),

    T("e21", "easy", "finance", "select where",
      ["Show payments made by {pay_method}.",
       "List all {pay_method} payments.",
       "Which payments used {pay_method}?"],
      """SELECT payment_id, order_id, amount, payment_method FROM payments
         WHERE payment_method = '{pay_method}'"""),

    T("e22", "easy", "logistics", "select where",
      ["Show shipments handled by {carrier}.",
       "List the shipments carried by {carrier}.",
       "Which shipments went through {carrier}?"],
      """SELECT shipment_id, order_id, carrier, status FROM shipments
         WHERE carrier = '{carrier}'"""),

    T("e23", "easy", "logistics", "select where",
      ["Show shipments with status {ship_status}.",
       "List all {ship_status} shipments.",
       "Which shipments are marked {ship_status}?"],
      """SELECT shipment_id, order_id, status FROM shipments
         WHERE status = '{ship_status}'"""),

    T("e24", "easy", "sales", "select where date order_by",
      ["Show customers who signed up after {signup_year}.",
       "List customers joining later than {signup_year}.",
       "Which customers registered after the year {signup_year}?"],
      """SELECT customer_id, customer_name, signup_date FROM customers
         WHERE EXTRACT(YEAR FROM signup_date) > {signup_year}
         ORDER BY signup_date"""),
]


# ===========================================================================
# MEDIUM — GROUP BY, HAVING, COUNT, SUM, AVG
# ===========================================================================

MEDIUM: list[Template] = [
    T("m01", "medium", "sales", "count where",
      ["How many customers are from {country}?",
       "Count the customers based in {country}.",
       "What is the number of customers in {country}?"],
      """SELECT COUNT(*) FROM customers WHERE country = '{country}'"""),

    T("m02", "medium", "sales", "count group_by order_by",
      ["How many customers do we have in each country?",
       "Count customers per country.",
       "Show the customer count broken down by country."],
      """SELECT country, COUNT(*) AS customer_count FROM customers
         GROUP BY country ORDER BY customer_count DESC"""),

    T("m03", "medium", "sales", "count group_by",
      ["How many customers are in each segment?",
       "Count customers by segment.",
       "Show the split of customers between segments."],
      """SELECT customer_segment, COUNT(*) AS customer_count FROM customers
         GROUP BY customer_segment ORDER BY customer_count DESC"""),

    T("m04", "medium", "sales", "count group_by order_by",
      ["How many orders are there in each status?",
       "Count orders by status.",
       "Show the number of orders per status."],
      """SELECT status, COUNT(*) AS order_count FROM orders
         GROUP BY status ORDER BY order_count DESC"""),

    T("m05", "medium", "sales", "sum group_by order_by",
      ["What is the total order value for each status?",
       "Show total revenue grouped by order status.",
       "Sum up order amounts by status."],
      """SELECT status, SUM(total_amount) AS total_value FROM orders
         GROUP BY status ORDER BY total_value DESC"""),

    T("m06", "medium", "sales", "avg",
      ["What is the average order value?",
       "Calculate the mean order total.",
       "On average, how much is an order worth?"],
      """SELECT AVG(total_amount) AS average_order_value FROM orders"""),

    T("m07", "medium", "sales", "count sum avg",
      ["Give me the total, average and count of all orders.",
       "Summarise our orders: how many, what total value, what average?",
       "Show order count, total value and average value."],
      """SELECT COUNT(*) AS order_count, SUM(total_amount) AS total_value,
                AVG(total_amount) AS average_value FROM orders"""),

    T("m08", "medium", "sales", "sum group_by date",
      ["What is the total revenue for each year?",
       "Show yearly revenue totals.",
       "Break down total order value by year."],
      """SELECT EXTRACT(YEAR FROM order_date) AS year,
                SUM(total_amount) AS revenue FROM orders
         GROUP BY year ORDER BY year"""),

    T("m09", "medium", "sales", "count group_by date",
      ["How many orders were placed each month of {year}?",
       "Show the monthly order count for {year}.",
       "Count orders by month in {year}."],
      """SELECT EXTRACT(MONTH FROM order_date) AS month, COUNT(*) AS order_count
         FROM orders WHERE EXTRACT(YEAR FROM order_date) = {year}
         GROUP BY month ORDER BY month"""),

    T("m10", "medium", "sales", "count group_by having",
      ["Which countries have more than {country_customer_count} customers?",
       "Show countries with a customer count above {country_customer_count}.",
       "List countries where we have over {country_customer_count} customers."],
      """SELECT country, COUNT(*) AS customer_count FROM customers
         GROUP BY country HAVING COUNT(*) > {country_customer_count}
         ORDER BY customer_count DESC"""),

    T("m11", "medium", "sales", "sum group_by having",
      ["Which customers have spent more than {customer_spend} in total?",
       "Show customers whose total spending exceeds {customer_spend}.",
       "List customers with lifetime order value above {customer_spend}."],
      """SELECT customer_id, SUM(total_amount) AS total_spent FROM orders
         GROUP BY customer_id HAVING SUM(total_amount) > {customer_spend}
         ORDER BY total_spent DESC"""),

    T("m12", "medium", "sales", "count group_by having",
      ["Which customers have placed more than {customer_order_count} orders?",
       "Show customers with over {customer_order_count} orders.",
       "List repeat customers with more than {customer_order_count} orders."],
      """SELECT customer_id, COUNT(*) AS order_count FROM orders
         GROUP BY customer_id HAVING COUNT(*) > {customer_order_count}
         ORDER BY order_count DESC"""),

    T("m13", "medium", "catalogue", "count group_by",
      ["How many products are in each category?",
       "Count products per category id.",
       "Show the number of products for every category."],
      """SELECT category_id, COUNT(*) AS product_count FROM products
         GROUP BY category_id ORDER BY product_count DESC"""),

    T("m14", "medium", "catalogue", "avg group_by",
      ["What is the average product price in each category?",
       "Show mean unit price by category id.",
       "Calculate average price per category."],
      """SELECT category_id, AVG(unit_price) AS average_price FROM products
         GROUP BY category_id ORDER BY average_price DESC"""),

    T("m15", "medium", "catalogue", "min max avg",
      ["What are the cheapest, most expensive and average product prices?",
       "Show the minimum, maximum and mean product price.",
       "Give me price statistics across all products."],
      """SELECT MIN(unit_price) AS cheapest, MAX(unit_price) AS most_expensive,
                AVG(unit_price) AS average_price FROM products"""),

    T("m16", "medium", "hr", "sum group_by",
      ["What is the total salary cost per department?",
       "Show total payroll by department id.",
       "Sum employee salaries for each department."],
      """SELECT department_id, SUM(salary) AS total_salary FROM employees
         GROUP BY department_id ORDER BY total_salary DESC"""),

    T("m17", "medium", "hr", "avg group_by order_by",
      ["What is the average salary for each job title?",
       "Show mean pay by job title.",
       "Calculate the average salary per role."],
      """SELECT job_title, AVG(salary) AS average_salary FROM employees
         GROUP BY job_title ORDER BY average_salary DESC"""),

    T("m18", "medium", "hr", "count group_by",
      ["How many employees work in each department?",
       "Count staff per department id.",
       "Show headcount by department."],
      """SELECT department_id, COUNT(*) AS employee_count FROM employees
         GROUP BY department_id ORDER BY employee_count DESC"""),

    T("m19", "medium", "finance", "sum group_by order_by",
      ["What is the total amount paid by each payment method?",
       "Show payment totals grouped by method.",
       "Sum payments per payment method."],
      """SELECT payment_method, SUM(amount) AS total_paid FROM payments
         GROUP BY payment_method ORDER BY total_paid DESC"""),

    T("m20", "medium", "finance", "count group_by",
      ["How many payments are there in each status?",
       "Count payments by status.",
       "Show the breakdown of payment statuses."],
      """SELECT status, COUNT(*) AS payment_count FROM payments
         GROUP BY status ORDER BY payment_count DESC"""),

    T("m21", "medium", "logistics", "count group_by",
      ["How many shipments does each carrier handle?",
       "Count shipments per carrier.",
       "Show shipment volume by carrier."],
      """SELECT carrier, COUNT(*) AS shipment_count FROM shipments
         GROUP BY carrier ORDER BY shipment_count DESC"""),

    T("m22", "medium", "logistics", "sum group_by",
      ["What is the total stock held in each warehouse?",
       "Show total inventory quantity per warehouse id.",
       "Sum stock on hand by warehouse."],
      """SELECT warehouse_id, SUM(quantity_on_hand) AS total_stock FROM inventory
         GROUP BY warehouse_id ORDER BY total_stock DESC"""),

    T("m23", "medium", "sales", "sum group_by having order_by",
      ["Which products have been ordered more than {product_total_qty} units in total?",
       "Show products with total ordered quantity above {product_total_qty}.",
       "List products where more than {product_total_qty} units have been sold."],
      """SELECT product_id, SUM(quantity) AS total_quantity FROM order_items
         GROUP BY product_id HAVING SUM(quantity) > {product_total_qty}
         ORDER BY total_quantity DESC"""),

    T("m24", "medium", "catalogue", "avg group_by",
      ["What is the average supplier rating per country?",
       "Show mean supplier rating grouped by country.",
       "Calculate average rating of suppliers in each country."],
      """SELECT country, AVG(rating) AS average_rating FROM suppliers
         GROUP BY country ORDER BY average_rating DESC NULLS LAST"""),

    T("m25", "medium", "sales", "count where date",
      ["How many orders were placed in {year}?",
       "Count the orders from {year}.",
       "What was our order volume in {year}?"],
      """SELECT COUNT(*) FROM orders WHERE EXTRACT(YEAR FROM order_date) = {year}"""),

    T("m26", "medium", "sales", "avg group_by",
      ["What is the average credit limit for each customer segment?",
       "Show mean credit limit by segment.",
       "Calculate average credit limit per segment."],
      """SELECT customer_segment, AVG(credit_limit) AS average_credit_limit
         FROM customers GROUP BY customer_segment"""),
]


# ===========================================================================
# HARD — JOIN, multiple JOINs, subqueries, CTEs
# ===========================================================================

HARD: list[Template] = [
    T("h01", "hard", "sales", "join count group_by order_by",
      ["Show each customer's name and how many orders they placed.",
       "List customers with their order counts.",
       "How many orders has each customer made? Include their name."],
      """SELECT c.customer_name, COUNT(o.order_id) AS order_count
         FROM customers c JOIN orders o ON o.customer_id = c.customer_id
         GROUP BY c.customer_id, c.customer_name ORDER BY order_count DESC"""),

    T("h02", "hard", "sales", "multi_join sum group_by order_by limit",
      ["Who are the top {n} customers by revenue?",
       "Show the {n} highest spending customers.",
       "List our {n} biggest customers by total sales value."],
      """SELECT c.customer_name,
                SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue
         FROM customers c
         JOIN orders o ON o.customer_id = c.customer_id
         JOIN order_items oi ON oi.order_id = o.order_id
         GROUP BY c.customer_id, c.customer_name
         ORDER BY revenue DESC LIMIT {n}"""),

    T("h03", "hard", "catalogue", "join where",
      ["Show all products in the {category} category.",
       "List products belonging to {category}.",
       "Which products are categorised as {category}?"],
      """SELECT p.product_name, p.unit_price
         FROM products p JOIN categories c ON p.category_id = c.category_id
         WHERE c.category_name = '{category}'"""),

    T("h04", "hard", "catalogue", "join",
      ["Show each product with the name of its supplier.",
       "List products alongside their supplier names.",
       "Which supplier provides each product?"],
      """SELECT p.product_name, s.supplier_name
         FROM products p JOIN suppliers s ON p.supplier_id = s.supplier_id
         ORDER BY s.supplier_name, p.product_name"""),

    T("h05", "hard", "catalogue", "join count group_by order_by",
      ["How many products does each supplier provide?",
       "Count products per supplier, showing the supplier name.",
       "Show suppliers with their product counts."],
      """SELECT s.supplier_name, COUNT(p.product_id) AS product_count
         FROM suppliers s JOIN products p ON p.supplier_id = s.supplier_id
         GROUP BY s.supplier_id, s.supplier_name ORDER BY product_count DESC"""),

    T("h06", "hard", "sales", "multi_join sum group_by order_by",
      ["What is the total revenue for each product category?",
       "Show revenue broken down by category name.",
       "Which categories generate the most sales revenue?"],
      """SELECT c.category_name,
                SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue
         FROM categories c
         JOIN products p ON p.category_id = c.category_id
         JOIN order_items oi ON oi.product_id = p.product_id
         GROUP BY c.category_id, c.category_name ORDER BY revenue DESC"""),

    T("h07", "hard", "hr", "join",
      ["Show each employee with their department name.",
       "List employees and the departments they work in.",
       "Which department does each employee belong to?"],
      """SELECT e.first_name, e.last_name, d.department_name
         FROM employees e JOIN departments d ON e.department_id = d.department_id
         ORDER BY d.department_name, e.last_name"""),

    T("h08", "hard", "hr", "self_join",
      ["Show each employee alongside their manager's name.",
       "List employees with the name of the person they report to.",
       "Who manages each employee?"],
      """SELECT e.first_name AS employee_first_name, e.last_name AS employee_last_name,
                m.first_name AS manager_first_name, m.last_name AS manager_last_name
         FROM employees e JOIN employees m ON e.manager_id = m.employee_id
         ORDER BY m.last_name, e.last_name"""),

    T("h09", "hard", "sales", "multi_join count_distinct group_by having",
      ["Which customers bought products from more than {customer_category_count} categories?",
       "Show customers purchasing across more than {customer_category_count} different categories.",
       "List customers whose orders span over {customer_category_count} product categories."],
      """SELECT c.customer_name, COUNT(DISTINCT p.category_id) AS category_count
         FROM customers c
         JOIN orders o ON o.customer_id = c.customer_id
         JOIN order_items oi ON oi.order_id = o.order_id
         JOIN products p ON p.product_id = oi.product_id
         GROUP BY c.customer_id, c.customer_name
         HAVING COUNT(DISTINCT p.category_id) > {customer_category_count}
         ORDER BY category_count DESC"""),

    T("h10", "hard", "catalogue", "join sum group_by order_by limit",
      ["What are the top {n} best selling products by revenue?",
       "Show the {n} products generating the most revenue.",
       "List our {n} highest earning products."],
      """SELECT p.product_name,
                SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue
         FROM products p JOIN order_items oi ON oi.product_id = p.product_id
         GROUP BY p.product_id, p.product_name ORDER BY revenue DESC LIMIT {n}"""),

    T("h11", "hard", "sales", "left_join null_check",
      ["Which customers have never placed an order?",
       "Show customers with no orders at all.",
       "List all customers who have not ordered anything."],
      """SELECT c.customer_id, c.customer_name
         FROM customers c LEFT JOIN orders o ON o.customer_id = c.customer_id
         WHERE o.order_id IS NULL"""),

    T("h12", "hard", "catalogue", "subquery not_exists",
      ["Which products have never been ordered?",
       "Show products that have never appeared in an order.",
       "List items with no sales at all."],
      """SELECT p.product_id, p.product_name FROM products p
         WHERE NOT EXISTS (
             SELECT 1 FROM order_items oi WHERE oi.product_id = p.product_id)"""),

    T("h13", "hard", "sales", "subquery avg",
      ["Which customers spent more than the average customer?",
       "Show customers whose total spending is above average.",
       "List above-average spenders."],
      """SELECT customer_id, SUM(total_amount) AS total_spent FROM orders
         GROUP BY customer_id
         HAVING SUM(total_amount) > (
             SELECT AVG(customer_total) FROM (
                 SELECT SUM(total_amount) AS customer_total FROM orders
                 GROUP BY customer_id) t)
         ORDER BY total_spent DESC"""),

    T("h14", "hard", "catalogue", "correlated_subquery",
      ["Which products cost more than the average price in their own category?",
       "Show products priced above their category average.",
       "List items more expensive than the typical product in their category."],
      """SELECT p.product_name, p.unit_price, p.category_id FROM products p
         WHERE p.unit_price > (
             SELECT AVG(p2.unit_price) FROM products p2
             WHERE p2.category_id = p.category_id)
         ORDER BY p.unit_price DESC"""),

    T("h15", "hard", "hr", "correlated_subquery",
      ["Which employees earn more than the average salary in their department?",
       "Show staff paid above their department's average.",
       "List employees earning more than their department average."],
      """SELECT e.first_name, e.last_name, e.salary, e.department_id FROM employees e
         WHERE e.salary > (
             SELECT AVG(e2.salary) FROM employees e2
             WHERE e2.department_id = e.department_id)
         ORDER BY e.salary DESC"""),

    T("h16", "hard", "sales", "cte join sum order_by limit",
      ["Using a CTE, show the top {n} categories by revenue.",
       "Which {n} categories earn the most? Use a common table expression.",
       "Show the {n} highest earning categories."],
      """WITH category_revenue AS (
             SELECT p.category_id,
                    SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue
             FROM order_items oi JOIN products p ON p.product_id = oi.product_id
             GROUP BY p.category_id)
         SELECT c.category_name, cr.revenue
         FROM category_revenue cr JOIN categories c ON c.category_id = cr.category_id
         ORDER BY cr.revenue DESC LIMIT {n}"""),

    T("h17", "hard", "sales", "join where date group_by",
      ["How much revenue did customers from {country} generate in {year}?",
       "Show total {year} sales for customers based in {country}.",
       "What did {country} customers spend during {year}?"],
      """SELECT SUM(o.total_amount) AS revenue
         FROM orders o JOIN customers c ON c.customer_id = o.customer_id
         WHERE c.country = '{country}'
           AND EXTRACT(YEAR FROM o.order_date) = {year}"""),

    T("h18", "hard", "logistics", "join count group_by order_by",
      ["How many shipments does each warehouse send?",
       "Show shipment counts per warehouse name.",
       "Which warehouses dispatch the most shipments?"],
      """SELECT w.warehouse_name, COUNT(s.shipment_id) AS shipment_count
         FROM warehouses w JOIN shipments s ON s.warehouse_id = w.warehouse_id
         GROUP BY w.warehouse_id, w.warehouse_name ORDER BY shipment_count DESC"""),

    T("h19", "hard", "logistics", "multi_join where",
      ["Which products are stored in the {warehouse_city} warehouses?",
       "Show items stocked in warehouses located in {warehouse_city}.",
       "List products held at {warehouse_city}."],
      """SELECT DISTINCT p.product_name, w.warehouse_name
         FROM products p
         JOIN inventory i ON i.product_id = p.product_id
         JOIN warehouses w ON w.warehouse_id = i.warehouse_id
         WHERE w.city = '{warehouse_city}'"""),

    T("h20", "hard", "catalogue", "join count group_by having",
      ["Which products are stocked in more than {product_warehouse_count} warehouses?",
       "Show items held across more than {product_warehouse_count} warehouses.",
       "List products stored in over {product_warehouse_count} locations."],
      """SELECT p.product_name, COUNT(DISTINCT i.warehouse_id) AS warehouse_count
         FROM products p JOIN inventory i ON i.product_id = p.product_id
         GROUP BY p.product_id, p.product_name
         HAVING COUNT(DISTINCT i.warehouse_id) > {product_warehouse_count}
         ORDER BY warehouse_count DESC"""),

    T("h21", "hard", "catalogue", "left_join null_check",
      ["Which categories contain no products?",
       "Show empty categories.",
       "List categories that have nothing assigned to them."],
      """SELECT c.category_id, c.category_name
         FROM categories c LEFT JOIN products p ON p.category_id = c.category_id
         WHERE p.product_id IS NULL"""),

    T("h22", "hard", "sales", "join sum group_by order_by limit",
      ["Which {n} sales representatives generated the most revenue?",
       "Show the top {n} employees by order value handled.",
       "List the {n} best performing sales reps by revenue."],
      """SELECT e.first_name, e.last_name, SUM(o.total_amount) AS revenue
         FROM employees e JOIN orders o ON o.employee_id = e.employee_id
         GROUP BY e.employee_id, e.first_name, e.last_name
         ORDER BY revenue DESC LIMIT {n}"""),

    T("h23", "hard", "finance", "join group_by having",
      ["Which orders have not been fully paid?",
       "Show orders where payments total less than the order value.",
       "List underpaid orders."],
      """SELECT o.order_id, o.total_amount, SUM(p.amount) AS paid
         FROM orders o JOIN payments p ON p.order_id = o.order_id
         WHERE p.status = 'completed'
         GROUP BY o.order_id, o.total_amount
         HAVING SUM(p.amount) < o.total_amount
         ORDER BY o.order_id"""),

    T("h24", "hard", "catalogue", "join where group_by",
      ["How many products does each supplier in {supplier_country} provide?",
       "Show product counts for suppliers based in {supplier_country}.",
       "Count products by supplier for {supplier_country}."],
      """SELECT s.supplier_name, COUNT(p.product_id) AS product_count
         FROM suppliers s JOIN products p ON p.supplier_id = s.supplier_id
         WHERE s.country = '{supplier_country}'
         GROUP BY s.supplier_id, s.supplier_name ORDER BY product_count DESC"""),
]


# ===========================================================================
# VERY HARD — window functions, nested aggregation, date comparison,
#             conditional aggregation
# ===========================================================================

VERY_HARD: list[Template] = [
    T("v01", "very_hard", "sales", "window rank join",
      ["Rank customers by total revenue.",
       "Show customers with a revenue ranking.",
       "Produce a ranked list of customers by how much they have spent."],
      """SELECT c.customer_name, SUM(o.total_amount) AS revenue,
                RANK() OVER (ORDER BY SUM(o.total_amount) DESC) AS revenue_rank
         FROM customers c JOIN orders o ON o.customer_id = c.customer_id
         GROUP BY c.customer_id, c.customer_name"""),

    T("v02", "very_hard", "catalogue", "window partition row_number",
      ["Show the top {small_n} products by revenue within each category.",
       "For every category, list its {small_n} best selling products.",
       "What are the {small_n} highest earning products per category?"],
      """WITH ranked AS (
             SELECT p.category_id, p.product_name,
                    SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.category_id
                        ORDER BY SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) DESC
                    ) AS rn
             FROM products p JOIN order_items oi ON oi.product_id = p.product_id
             GROUP BY p.category_id, p.product_id, p.product_name)
         SELECT category_id, product_name, revenue FROM ranked
         WHERE rn <= {small_n} ORDER BY category_id, revenue DESC"""),

    T("v03", "very_hard", "sales", "window running_total date",
      ["Show the running total of revenue by month for {year}.",
       "Calculate cumulative monthly revenue during {year}.",
       "What is the month-by-month cumulative sales total in {year}?"],
      """SELECT DATE_TRUNC('month', order_date) AS month,
                SUM(total_amount) AS monthly_revenue,
                SUM(SUM(total_amount)) OVER (
                    ORDER BY DATE_TRUNC('month', order_date)) AS running_total
         FROM orders WHERE EXTRACT(YEAR FROM order_date) = {year}
         GROUP BY month ORDER BY month"""),

    T("v04", "very_hard", "sales", "window lag date",
      ["Show month-over-month revenue change for {year}.",
       "Compare each month's revenue with the previous month in {year}.",
       "What is the monthly revenue difference across {year}?"],
      """SELECT DATE_TRUNC('month', order_date) AS month,
                SUM(total_amount) AS revenue,
                SUM(total_amount) - LAG(SUM(total_amount)) OVER (
                    ORDER BY DATE_TRUNC('month', order_date)) AS change_from_previous
         FROM orders WHERE EXTRACT(YEAR FROM order_date) = {year}
         GROUP BY month ORDER BY month"""),

    T("v05", "very_hard", "sales", "conditional_aggregation group_by",
      ["For each country, how many orders were delivered versus cancelled?",
       "Show delivered and cancelled order counts per country.",
       "Break down delivered vs cancelled orders by customer country."],
      """SELECT c.country,
                COUNT(*) FILTER (WHERE o.status = 'delivered') AS delivered,
                COUNT(*) FILTER (WHERE o.status = 'cancelled') AS cancelled
         FROM customers c JOIN orders o ON o.customer_id = c.customer_id
         GROUP BY c.country ORDER BY c.country"""),

    T("v06", "very_hard", "sales", "conditional_aggregation percentage",
      ["What percentage of orders were delivered in each country?",
       "Show the delivery rate per country as a percentage.",
       "Calculate the share of orders delivered, by country."],
      """SELECT c.country,
                ROUND(100.0 * COUNT(*) FILTER (WHERE o.status = 'delivered')
                      / COUNT(*), 2) AS delivered_pct
         FROM customers c JOIN orders o ON o.customer_id = c.customer_id
         GROUP BY c.country ORDER BY delivered_pct DESC"""),

    T("v07", "very_hard", "logistics", "date_diff avg",
      ["What is the average number of days between order and delivery?",
       "How long does delivery take on average?",
       "Calculate mean delivery time in days."],
      """SELECT AVG(EXTRACT(EPOCH FROM (s.delivered_date - o.order_date)) / 86400.0)
                AS avg_delivery_days
         FROM orders o JOIN shipments s ON s.order_id = o.order_id
         WHERE s.delivered_date IS NOT NULL"""),

    T("v08", "very_hard", "logistics", "date_diff group_by avg",
      ["Which carrier delivers fastest on average?",
       "Show mean delivery days per carrier.",
       "Compare carriers by average delivery time."],
      """SELECT s.carrier,
                AVG(EXTRACT(EPOCH FROM (s.delivered_date - s.shipped_date)) / 86400.0)
                AS avg_days
         FROM shipments s
         WHERE s.delivered_date IS NOT NULL AND s.shipped_date IS NOT NULL
         GROUP BY s.carrier ORDER BY avg_days ASC"""),

    T("v09", "very_hard", "sales", "window share_of_total",
      ["What share of total revenue does each category represent?",
       "Show each category's percentage of overall revenue.",
       "Calculate revenue share by category."],
      """SELECT c.category_name,
                SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue,
                ROUND(100.0 * SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100))
                      / SUM(SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)))
                        OVER (), 2) AS pct_of_total
         FROM categories c
         JOIN products p ON p.category_id = c.category_id
         JOIN order_items oi ON oi.product_id = p.product_id
         GROUP BY c.category_id, c.category_name ORDER BY revenue DESC"""),

    T("v10", "very_hard", "catalogue", "recursive_cte hierarchy",
      ["List every subcategory beneath {top_category}, at any depth.",
       "Show the full category tree under {top_category}.",
       "What categories fall under {top_category}, including nested ones?"],
      """WITH RECURSIVE tree AS (
             SELECT category_id, category_name, parent_category_id, 0 AS depth
             FROM categories WHERE category_name = '{top_category}'
             UNION ALL
             SELECT c.category_id, c.category_name, c.parent_category_id, t.depth + 1
             FROM categories c JOIN tree t ON c.parent_category_id = t.category_id)
         SELECT category_id, category_name, depth FROM tree
         WHERE depth > 0 ORDER BY depth, category_name"""),

    T("v11", "very_hard", "hr", "recursive_cte hierarchy",
      ["Show the reporting depth of every employee in the org chart.",
       "How many levels below the CEO is each employee?",
       "Produce the organisational hierarchy with depth levels."],
      """WITH RECURSIVE org AS (
             SELECT employee_id, first_name, last_name, manager_id, 0 AS level
             FROM employees WHERE manager_id IS NULL
             UNION ALL
             SELECT e.employee_id, e.first_name, e.last_name, e.manager_id, o.level + 1
             FROM employees e JOIN org o ON e.manager_id = o.employee_id)
         SELECT employee_id, first_name, last_name, level FROM org
         ORDER BY level, last_name"""),

    T("v12", "very_hard", "sales", "nested_aggregation",
      ["What is the average number of items per order?",
       "On average, how many line items does an order contain?",
       "Calculate the mean order size in line items."],
      """SELECT AVG(item_count) AS avg_items_per_order FROM (
             SELECT order_id, COUNT(*) AS item_count FROM order_items
             GROUP BY order_id) t"""),

    T("v13", "very_hard", "sales", "percentile",
      ["What is the median order value?",
       "Find the 50th percentile of order totals.",
       "Calculate the median value of an order."],
      """SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_amount)
                AS median_order_value FROM orders"""),

    T("v14", "very_hard", "sales", "window ntile",
      ["Divide customers into 4 quartiles by total spending.",
       "Show which spending quartile each customer falls into.",
       "Assign customers to quartiles based on revenue."],
      """SELECT customer_id, SUM(total_amount) AS total_spent,
                NTILE(4) OVER (ORDER BY SUM(total_amount) DESC) AS spending_quartile
         FROM orders GROUP BY customer_id ORDER BY total_spent DESC"""),

    T("v15", "very_hard", "logistics", "date_comparison",
      ["Which orders were shipped after their required date?",
       "Show late shipments compared to the required date.",
       "List orders that missed their required delivery date."],
      """SELECT o.order_id, o.required_date, s.shipped_date
         FROM orders o JOIN shipments s ON s.order_id = o.order_id
         WHERE s.shipped_date IS NOT NULL
           AND s.shipped_date::date > o.required_date
         ORDER BY o.order_id"""),

    T("v16", "very_hard", "sales", "window first_value date",
      ["Show each customer's first order date and their total order count.",
       "When did each customer first order, and how many have they placed?",
       "List first order date per customer with their order volume."],
      """SELECT c.customer_name, MIN(o.order_date) AS first_order,
                COUNT(o.order_id) AS order_count
         FROM customers c JOIN orders o ON o.customer_id = c.customer_id
         GROUP BY c.customer_id, c.customer_name ORDER BY first_order"""),

    T("v17", "very_hard", "sales", "cohort date group_by",
      ["Show total revenue by customer signup year.",
       "Group revenue by the year each customer joined.",
       "What is the lifetime value of each signup cohort?"],
      """SELECT EXTRACT(YEAR FROM c.signup_date) AS cohort_year,
                COUNT(DISTINCT c.customer_id) AS customers,
                SUM(o.total_amount) AS revenue
         FROM customers c JOIN orders o ON o.customer_id = c.customer_id
         GROUP BY cohort_year ORDER BY cohort_year"""),

    T("v18", "very_hard", "sales", "conditional_aggregation group_by",
      ["For each year, show revenue from business versus individual customers.",
       "Compare business and individual revenue by year.",
       "Break down yearly revenue by customer segment."],
      """SELECT EXTRACT(YEAR FROM o.order_date) AS year,
                SUM(o.total_amount) FILTER (WHERE c.customer_segment = 'business')
                    AS business_revenue,
                SUM(o.total_amount) FILTER (WHERE c.customer_segment = 'individual')
                    AS individual_revenue
         FROM orders o JOIN customers c ON c.customer_id = o.customer_id
         GROUP BY year ORDER BY year"""),
]


# ===========================================================================
# ENTERPRISE — ambiguity, NULL handling, business rules, cross-domain
# ===========================================================================

ENTERPRISE: list[Template] = [
    T("x01", "enterprise", "sales", "business_rule where",
      ["What is our actual revenue, excluding cancelled and returned orders?",
       "Show real revenue after removing cancellations and returns.",
       "Calculate net revenue, ignoring cancelled and returned orders."],
      """SELECT SUM(total_amount) AS net_revenue FROM orders
         WHERE status NOT IN ('cancelled', 'returned')"""),

    T("x02", "enterprise", "sales", "null_handling left_join",
      ["Which orders have no sales representative assigned?",
       "Show self-service orders with no rep.",
       "List orders where no employee handled the sale."],
      """SELECT order_id, customer_id, order_date, total_amount FROM orders
         WHERE employee_id IS NULL"""),

    T("x03", "enterprise", "sales", "null_handling left_join group_by",
      ["Compare revenue from orders with a sales rep against those without one.",
       "How much revenue comes from assisted versus self-service orders?",
       "Split revenue by whether an employee handled the order."],
      """SELECT CASE WHEN employee_id IS NULL THEN 'self_service' ELSE 'assisted' END
                AS channel,
                COUNT(*) AS order_count, SUM(total_amount) AS revenue
         FROM orders GROUP BY channel"""),

    T("x04", "enterprise", "catalogue", "null_handling",
      ["Which products have no supplier on record?",
       "Show items where the supplier is unknown.",
       "List products missing supplier information."],
      """SELECT product_id, product_name FROM products WHERE supplier_id IS NULL"""),

    T("x05", "enterprise", "catalogue", "null_handling coalesce",
      ["Show suppliers that have not been rated yet.",
       "Which suppliers are missing a quality rating?",
       "List unrated suppliers."],
      """SELECT supplier_id, supplier_name, country FROM suppliers
         WHERE rating IS NULL"""),

    T("x06", "enterprise", "sales", "null_handling business_rule",
      ["Which business customers have a credit limit below {customer_credit_limit}?",
       "Show business accounts with credit limits under {customer_credit_limit}.",
       "List business customers whose credit limit is less than {customer_credit_limit}."],
      """SELECT customer_id, customer_name, credit_limit FROM customers
         WHERE customer_segment = 'business'
           AND credit_limit IS NOT NULL AND credit_limit < {customer_credit_limit}
         ORDER BY credit_limit"""),

    T("x07", "enterprise", "logistics", "null_handling date",
      ["Which orders have shipped but not yet been delivered?",
       "Show shipments still in transit.",
       "List consignments that left the warehouse but have not arrived."],
      """SELECT s.shipment_id, s.order_id, s.shipped_date, s.carrier
         FROM shipments s
         WHERE s.shipped_date IS NOT NULL AND s.delivered_date IS NULL
         ORDER BY s.shipped_date"""),

    T("x08", "enterprise", "logistics", "cross_domain business_rule join",
      ["Which products are below their reorder level in any warehouse?",
       "Show stock that needs reordering.",
       "List items whose quantity has fallen under the reorder threshold."],
      """SELECT p.product_name, w.warehouse_name,
                i.quantity_on_hand, i.reorder_level
         FROM inventory i
         JOIN products p ON p.product_id = i.product_id
         JOIN warehouses w ON w.warehouse_id = i.warehouse_id
         WHERE i.quantity_on_hand < i.reorder_level
         ORDER BY p.product_name"""),

    T("x09", "enterprise", "logistics", "cross_domain business_rule",
      ["Which discontinued products still have stock on hand?",
       "Show inventory held for products we no longer sell.",
       "List discontinued items that are still in the warehouse."],
      """SELECT p.product_name, SUM(i.quantity_on_hand) AS stock_remaining
         FROM products p JOIN inventory i ON i.product_id = p.product_id
         WHERE p.is_discontinued = TRUE
         GROUP BY p.product_id, p.product_name
         HAVING SUM(i.quantity_on_hand) > 0
         ORDER BY stock_remaining DESC"""),

    T("x10", "enterprise", "finance", "business_rule join group_by having",
      ["Which orders are still awaiting payment?",
       "Show unpaid orders that were not cancelled.",
       "List orders with no completed payment against them."],
      """SELECT o.order_id, o.total_amount, o.status FROM orders o
         WHERE o.status <> 'cancelled'
           AND NOT EXISTS (
               SELECT 1 FROM payments p
               WHERE p.order_id = o.order_id AND p.status = 'completed')
         ORDER BY o.order_id"""),

    T("x11", "enterprise", "cross_domain", "cross_domain multi_join group_by",
      ["Which department generates the most sales revenue?",
       "Show revenue attributed to each department through its sales reps.",
       "Break down sales by the department of the handling employee."],
      """SELECT d.department_name, SUM(o.total_amount) AS revenue
         FROM orders o
         JOIN employees e ON e.employee_id = o.employee_id
         JOIN departments d ON d.department_id = e.department_id
         GROUP BY d.department_id, d.department_name ORDER BY revenue DESC"""),

    T("x12", "enterprise", "cross_domain", "cross_domain multi_join where",
      ["Which customers in {country} bought products supplied from {supplier_country}?",
       "Show {country} customers purchasing goods sourced in {supplier_country}.",
       "List buyers in {country} of products from {supplier_country} suppliers."],
      """SELECT DISTINCT c.customer_name, s.supplier_name
         FROM customers c
         JOIN orders o ON o.customer_id = c.customer_id
         JOIN order_items oi ON oi.order_id = o.order_id
         JOIN products p ON p.product_id = oi.product_id
         JOIN suppliers s ON s.supplier_id = p.supplier_id
         WHERE c.country = '{country}' AND s.country = '{supplier_country}'"""),

    T("x13", "enterprise", "catalogue", "business_rule margin null_handling",
      ["Which products have the highest profit margin?",
       "Show items ranked by margin between price and cost.",
       "List the most profitable products by margin percentage."],
      """SELECT product_name, unit_price, unit_cost,
                ROUND(100.0 * (unit_price - unit_cost) / NULLIF(unit_price, 0), 2)
                AS margin_pct
         FROM products
         WHERE unit_cost IS NOT NULL AND unit_price > 0
         ORDER BY margin_pct DESC"""),

    T("x14", "enterprise", "sales", "business_rule date churn",
      ["Which customers have not ordered in the last {churn_months} months?",
       "Show customers inactive for over {churn_months} months.",
       "List churned customers with no order in {churn_months} months."],
      """SELECT c.customer_id, c.customer_name, MAX(o.order_date) AS last_order
         FROM customers c JOIN orders o ON o.customer_id = c.customer_id
         GROUP BY c.customer_id, c.customer_name
         HAVING MAX(o.order_date) < {as_of} - INTERVAL '{churn_months} months'
         ORDER BY last_order"""),

    T("x15", "enterprise", "logistics", "cross_domain business_rule",
      ["How full is each warehouse relative to its capacity?",
       "Show warehouse utilisation as a percentage of capacity.",
       "Compare stock held against capacity for each warehouse."],
      """SELECT w.warehouse_name, w.capacity_units,
                COALESCE(SUM(i.quantity_on_hand), 0) AS stock_held,
                ROUND(100.0 * COALESCE(SUM(i.quantity_on_hand), 0)
                      / NULLIF(w.capacity_units, 0), 2) AS utilisation_pct
         FROM warehouses w LEFT JOIN inventory i ON i.warehouse_id = w.warehouse_id
         GROUP BY w.warehouse_id, w.warehouse_name, w.capacity_units
         ORDER BY utilisation_pct DESC"""),

    T("x16", "enterprise", "sales", "ambiguous business_rule",
      ["Who are our best customers?",
       "Show our most valuable customers.",
       "Which customers matter most to the business?"],
      """SELECT c.customer_name, COUNT(DISTINCT o.order_id) AS order_count,
                SUM(o.total_amount) AS total_revenue
         FROM customers c JOIN orders o ON o.customer_id = c.customer_id
         WHERE o.status NOT IN ('cancelled', 'returned')
         GROUP BY c.customer_id, c.customer_name
         ORDER BY total_revenue DESC LIMIT 20"""),

    T("x17", "enterprise", "sales", "ambiguous business_rule",
      ["How is the business performing this year compared to last year?",
       "Compare this year's revenue against last year.",
       "Show year-on-year revenue performance."],
      """SELECT EXTRACT(YEAR FROM order_date) AS year,
                COUNT(*) AS order_count, SUM(total_amount) AS revenue
         FROM orders
         WHERE status NOT IN ('cancelled', 'returned')
           AND order_date >= DATE_TRUNC('year', {as_of}) - INTERVAL '1 year'
         GROUP BY year ORDER BY year"""),

    T("x18", "enterprise", "cross_domain", "cross_domain multi_join business_rule",
      ["Which sales representatives handled orders that were later returned?",
       "Show employees associated with returned orders.",
       "List reps whose orders came back as returns."],
      """SELECT e.first_name, e.last_name, COUNT(*) AS returned_orders
         FROM orders o JOIN employees e ON e.employee_id = o.employee_id
         WHERE o.status = 'returned'
         GROUP BY e.employee_id, e.first_name, e.last_name
         ORDER BY returned_orders DESC"""),

    T("x19", "enterprise", "finance", "business_rule null_handling",
      ["What proportion of order value has actually been collected?",
       "Show collected payments against total order value.",
       "How much of what we billed has been paid?"],
      """SELECT SUM(o.total_amount) AS billed,
                COALESCE(SUM(paid.total_paid), 0) AS collected,
                ROUND(100.0 * COALESCE(SUM(paid.total_paid), 0)
                      / NULLIF(SUM(o.total_amount), 0), 2) AS collected_pct
         FROM orders o
         LEFT JOIN (
             SELECT order_id, SUM(amount) AS total_paid FROM payments
             WHERE status = 'completed' GROUP BY order_id) paid
           ON paid.order_id = o.order_id
         WHERE o.status <> 'cancelled'"""),

    T("x20", "enterprise", "cross_domain", "cross_domain business_rule date",
      ["Which carriers most often deliver later than the required date?",
       "Show late delivery counts by carrier.",
       "Compare carriers on missed delivery deadlines."],
      """SELECT s.carrier,
                COUNT(*) FILTER (WHERE s.delivered_date::date > o.required_date)
                    AS late_deliveries,
                COUNT(*) AS total_delivered
         FROM shipments s JOIN orders o ON o.order_id = s.order_id
         WHERE s.delivered_date IS NOT NULL AND o.required_date IS NOT NULL
         GROUP BY s.carrier ORDER BY late_deliveries DESC"""),
]


# ===========================================================================
# SUPPLEMENTARY TEMPLATES
#
# The first pass left the benchmark skewed: sales had ~30x the coverage of
# finance, and the medium and very_hard tiers were thin. A benchmark that
# barely touches payments or inventory cannot detect a model that is bad at
# them, so these fill the gaps rather than adding more of what already existed.
# ===========================================================================

EASY_EXTRA: list[Template] = [
    T("e25", "easy", "finance", "select where",
      ["Show payments larger than {payment_amount} dollars.",
       "List payments above {payment_amount}.",
       "Which payments exceeded {payment_amount}?"],
      """SELECT payment_id, order_id, amount, payment_method FROM payments
         WHERE amount > {payment_amount}"""),

    T("e26", "easy", "finance", "select where date",
      ["Show payments made in {year}.",
       "List all payments received during {year}.",
       "Which payments were recorded in {year}?"],
      """SELECT payment_id, order_id, payment_date, amount FROM payments
         WHERE EXTRACT(YEAR FROM payment_date) = {year}"""),

    T("e27", "easy", "finance", "select where",
      ["Show payments with status {pay_status}.",
       "List all {pay_status} payments.",
       "Which payments are marked {pay_status}?"],
      """SELECT payment_id, order_id, amount, status FROM payments
         WHERE status = '{pay_status}'"""),

    T("e28", "easy", "logistics", "select where date",
      ["Show shipments dispatched in {year}.",
       "List shipments that left in {year}.",
       "Which consignments shipped during {year}?"],
      """SELECT shipment_id, order_id, shipped_date, carrier FROM shipments
         WHERE EXTRACT(YEAR FROM shipped_date) = {year}"""),

    T("e29", "easy", "hr", "select where date",
      ["Show employees hired after {hire_year}.",
       "List staff who joined later than {hire_year}.",
       "Which employees started after {hire_year}?"],
      """SELECT employee_id, first_name, last_name, hire_date FROM employees
         WHERE EXTRACT(YEAR FROM hire_date) > {hire_year} ORDER BY hire_date"""),

    T("e30", "easy", "hr", "select where boolean",
      ["Show employees who have left the company.",
       "List inactive staff members.",
       "Which employees are no longer active?"],
      """SELECT employee_id, first_name, last_name FROM employees
         WHERE is_active = FALSE"""),

    T("e31", "easy", "logistics", "select where",
      ["Show warehouses located in {warehouse_country}.",
       "List the warehouses in {warehouse_country}.",
       "Which warehouses do we operate in {warehouse_country}?"],
      """SELECT warehouse_id, warehouse_name, city FROM warehouses
         WHERE country = '{warehouse_country}'"""),

    T("e32", "easy", "catalogue", "select where",
      ["Show suppliers rated above {supplier_rating}.",
       "List suppliers with a rating better than {supplier_rating}.",
       "Which suppliers score higher than {supplier_rating}?"],
      """SELECT supplier_id, supplier_name, rating FROM suppliers
         WHERE rating > {supplier_rating} ORDER BY rating DESC"""),

    T("e33", "easy", "logistics", "select where",
      ["Show inventory rows with fewer than {inventory_qty} units in stock.",
       "List stock entries below {inventory_qty} units.",
       "Which inventory records hold less than {inventory_qty} items?"],
      """SELECT inventory_id, product_id, warehouse_id, quantity_on_hand
         FROM inventory WHERE quantity_on_hand < {inventory_qty}"""),

    T("e34", "easy", "sales", "select where order_by",
      ["Show orders with shipping costs above {order_shipping} dollars.",
       "List orders where shipping exceeded {order_shipping}.",
       "Which orders had shipping charges over {order_shipping}?"],
      """SELECT order_id, shipping_cost, total_amount FROM orders
         WHERE shipping_cost > {order_shipping} ORDER BY shipping_cost DESC"""),
]

MEDIUM_EXTRA: list[Template] = [
    T("m27", "medium", "finance", "count where date",
      ["How many payments were received in {year}?",
       "Count the payments recorded during {year}.",
       "What was the payment volume in {year}?"],
      """SELECT COUNT(*) FROM payments
         WHERE EXTRACT(YEAR FROM payment_date) = {year}"""),

    T("m28", "medium", "finance", "sum group_by date",
      ["What is the total amount paid each year?",
       "Show yearly payment totals.",
       "Break down collected payments by year."],
      """SELECT EXTRACT(YEAR FROM payment_date) AS year, SUM(amount) AS total_paid
         FROM payments GROUP BY year ORDER BY year"""),

    T("m29", "medium", "finance", "avg group_by",
      ["What is the average payment amount for each method?",
       "Show mean payment value by payment method.",
       "Calculate the average transaction size per method."],
      """SELECT payment_method, AVG(amount) AS average_amount FROM payments
         GROUP BY payment_method ORDER BY average_amount DESC"""),

    T("m30", "medium", "finance", "count group_by having",
      ["Which payment methods were used more than {payment_method_uses} times?",
       "Show methods with over {payment_method_uses} transactions.",
       "List payment methods exceeding {payment_method_uses} uses."],
      """SELECT payment_method, COUNT(*) AS uses FROM payments
         GROUP BY payment_method HAVING COUNT(*) > {payment_method_uses}
         ORDER BY uses DESC"""),

    T("m31", "medium", "logistics", "count group_by date",
      ["How many shipments were sent each year?",
       "Show yearly shipment counts.",
       "Break down shipment volume by year."],
      """SELECT EXTRACT(YEAR FROM shipped_date) AS year, COUNT(*) AS shipments
         FROM shipments WHERE shipped_date IS NOT NULL
         GROUP BY year ORDER BY year"""),

    T("m32", "medium", "logistics", "sum group_by having",
      ["Which products hold more than {product_total_stock} units across all warehouses?",
       "Show products with total stock above {product_total_stock}.",
       "List items where combined inventory exceeds {product_total_stock}."],
      """SELECT product_id, SUM(quantity_on_hand) AS total_stock FROM inventory
         GROUP BY product_id HAVING SUM(quantity_on_hand) > {product_total_stock}
         ORDER BY total_stock DESC"""),

    T("m33", "medium", "logistics", "avg group_by",
      ["What is the average warehouse capacity per country?",
       "Show mean capacity of warehouses by country.",
       "Calculate average storage capacity in each country."],
      """SELECT country, AVG(capacity_units) AS average_capacity FROM warehouses
         GROUP BY country ORDER BY average_capacity DESC"""),

    T("m34", "medium", "hr", "count group_by boolean",
      ["How many employees are active versus inactive?",
       "Count staff by active status.",
       "Show the split between current and former employees."],
      """SELECT is_active, COUNT(*) AS employee_count FROM employees
         GROUP BY is_active"""),

    # Phrasings must stay distinguishable from h28, which answers the same
    # business question but returns department NAMES via a join. Identical
    # question text mapping to different SQL gives execution-based scoring two
    # conflicting gold answers.
    T("m35", "medium", "hr", "avg group_by having",
      ["Which department ids have an average salary above {dept_avg_salary}?",
       "Show department ids where mean pay exceeds {dept_avg_salary}.",
       "List department ids whose average salary is over {dept_avg_salary}."],
      """SELECT department_id, AVG(salary) AS average_salary FROM employees
         GROUP BY department_id HAVING AVG(salary) > {dept_avg_salary}
         ORDER BY average_salary DESC"""),

    T("m36", "medium", "hr", "count group_by date",
      ["How many employees were hired each year?",
       "Show hiring counts by year.",
       "Break down recruitment by year."],
      """SELECT EXTRACT(YEAR FROM hire_date) AS year, COUNT(*) AS hires
         FROM employees GROUP BY year ORDER BY year"""),

    T("m37", "medium", "catalogue", "count group_by",
      ["How many suppliers are there in each country?",
       "Count suppliers per country.",
       "Show the supplier distribution by country."],
      """SELECT country, COUNT(*) AS supplier_count FROM suppliers
         GROUP BY country ORDER BY supplier_count DESC"""),

    T("m38", "medium", "sales", "sum group_by date",
      ["What is the total shipping cost collected each year?",
       "Show yearly shipping revenue.",
       "Break down shipping charges by year."],
      """SELECT EXTRACT(YEAR FROM order_date) AS year,
                SUM(shipping_cost) AS total_shipping
         FROM orders GROUP BY year ORDER BY year"""),

    T("m39", "medium", "sales", "count group_by having",
      ["Which orders contain more than {order_line_count} line items?",
       "Show orders with over {order_line_count} products on them.",
       "List orders having more than {order_line_count} items."],
      """SELECT order_id, COUNT(*) AS item_count FROM order_items
         GROUP BY order_id HAVING COUNT(*) > {order_line_count}
         ORDER BY item_count DESC"""),

    T("m40", "medium", "sales", "avg group_by",
      ["What is the average order value for each status?",
       "Show mean order total by status.",
       "Calculate the average order size per status."],
      """SELECT status, AVG(total_amount) AS average_value FROM orders
         GROUP BY status ORDER BY average_value DESC"""),

    T("m41", "medium", "sales", "avg where date",
      ["What was the average order value in {year}?",
       "Show the mean order total for {year}.",
       "How much was a typical order worth in {year}?"],
      """SELECT AVG(total_amount) AS average_order_value FROM orders
         WHERE EXTRACT(YEAR FROM order_date) = {year}"""),

    T("m42", "medium", "catalogue", "avg group_by",
      ["What is the average discount given on each product?",
       "Show mean discount percentage per product id.",
       "Calculate average discount by product."],
      """SELECT product_id, AVG(discount_pct) AS average_discount FROM order_items
         GROUP BY product_id ORDER BY average_discount DESC"""),
]

HARD_EXTRA: list[Template] = [
    T("h25", "hard", "finance", "multi_join sum group_by order_by",
      ["How much has been collected from customers in each country?",
       "Show total payments grouped by customer country.",
       "Which countries have paid us the most?"],
      """SELECT c.country, SUM(p.amount) AS total_paid
         FROM payments p
         JOIN orders o ON o.order_id = p.order_id
         JOIN customers c ON c.customer_id = o.customer_id
         WHERE p.status = 'completed'
         GROUP BY c.country ORDER BY total_paid DESC"""),

    T("h26", "hard", "logistics", "join count group_by",
      ["How many shipments originate from each warehouse country?",
       "Show shipment counts by the country of the dispatching warehouse.",
       "Which countries dispatch the most shipments?"],
      """SELECT w.country, COUNT(s.shipment_id) AS shipment_count
         FROM shipments s JOIN warehouses w ON w.warehouse_id = s.warehouse_id
         GROUP BY w.country ORDER BY shipment_count DESC"""),

    T("h27", "hard", "hr", "join avg group_by order_by",
      ["What is the average salary in each department?",
       "Show mean pay by department name.",
       "Which departments pay the most on average?"],
      """SELECT d.department_name, AVG(e.salary) AS average_salary
         FROM employees e JOIN departments d ON d.department_id = e.department_id
         GROUP BY d.department_id, d.department_name ORDER BY average_salary DESC"""),

    T("h28", "hard", "hr", "join avg group_by having",
      ["Which departments have an average salary above {dept_avg_salary}?",
       "Show departments by name where mean pay exceeds {dept_avg_salary}.",
       "List departments whose average salary is over {dept_avg_salary}."],
      """SELECT d.department_name, AVG(e.salary) AS average_salary
         FROM employees e JOIN departments d ON d.department_id = e.department_id
         GROUP BY d.department_id, d.department_name
         HAVING AVG(e.salary) > {dept_avg_salary} ORDER BY average_salary DESC"""),

    T("h29", "hard", "catalogue", "join count group_by order_by",
      ["How many products are in each named category?",
       "Show product counts by category name.",
       "Which categories contain the most products?"],
      """SELECT c.category_name, COUNT(p.product_id) AS product_count
         FROM categories c JOIN products p ON p.category_id = c.category_id
         GROUP BY c.category_id, c.category_name ORDER BY product_count DESC"""),

    T("h30", "hard", "catalogue", "join where",
      ["Which products come from suppliers in {supplier_country}?",
       "Show items sourced from {supplier_country}.",
       "List products supplied by {supplier_country} companies."],
      """SELECT p.product_name, s.supplier_name
         FROM products p JOIN suppliers s ON s.supplier_id = p.supplier_id
         WHERE s.country = '{supplier_country}'"""),

    T("h31", "hard", "sales", "join where count group_by",
      ["How many {status} orders does each country have?",
       "Show {status} order counts by customer country.",
       "Break down {status} orders by country."],
      """SELECT c.country, COUNT(*) AS order_count
         FROM orders o JOIN customers c ON c.customer_id = o.customer_id
         WHERE o.status = '{status}'
         GROUP BY c.country ORDER BY order_count DESC"""),

    T("h32", "hard", "logistics", "multi_join sum group_by order_by limit",
      ["Which {warehouse_limit} warehouses hold the most valuable stock?",
       "Show the top {warehouse_limit} warehouses by inventory value.",
       "List the {warehouse_limit} warehouses with the highest stock value."],
      """SELECT w.warehouse_name,
                SUM(i.quantity_on_hand * p.unit_price) AS stock_value
         FROM inventory i
         JOIN warehouses w ON w.warehouse_id = i.warehouse_id
         JOIN products p ON p.product_id = i.product_id
         GROUP BY w.warehouse_id, w.warehouse_name
         ORDER BY stock_value DESC LIMIT {warehouse_limit}"""),

    T("h33", "hard", "catalogue", "join avg group_by having",
      ["Which categories have an average product price above {category_avg_price}?",
       "Show categories where the mean price exceeds {category_avg_price}.",
       "List category names whose average price is over {category_avg_price}."],
      """SELECT c.category_name, AVG(p.unit_price) AS average_price
         FROM categories c JOIN products p ON p.category_id = c.category_id
         GROUP BY c.category_id, c.category_name
         HAVING AVG(p.unit_price) > {category_avg_price} ORDER BY average_price DESC"""),

    T("h34", "hard", "finance", "subquery not_exists",
      ["Which customers have placed orders but never paid?",
       "Show customers with orders and no payments at all.",
       "List buyers who have never made a payment."],
      """SELECT DISTINCT c.customer_id, c.customer_name
         FROM customers c JOIN orders o ON o.customer_id = c.customer_id
         WHERE NOT EXISTS (
             SELECT 1 FROM payments p WHERE p.order_id = o.order_id)"""),

    T("h35", "hard", "catalogue", "left_join null_check",
      ["Which suppliers provide no products?",
       "Show suppliers with nothing in our catalogue.",
       "List suppliers that have no products registered."],
      """SELECT s.supplier_id, s.supplier_name
         FROM suppliers s LEFT JOIN products p ON p.supplier_id = s.supplier_id
         WHERE p.product_id IS NULL"""),

    T("h36", "hard", "catalogue", "multi_join sum group_by order_by limit",
      ["Which {n} suppliers generate the most revenue?",
       "Show the top {n} suppliers by sales value.",
       "List our {n} highest earning suppliers."],
      """SELECT s.supplier_name,
                SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue
         FROM suppliers s
         JOIN products p ON p.supplier_id = s.supplier_id
         JOIN order_items oi ON oi.product_id = p.product_id
         GROUP BY s.supplier_id, s.supplier_name ORDER BY revenue DESC LIMIT {n}"""),
]

VERY_HARD_EXTRA: list[Template] = [
    T("v19", "very_hard", "catalogue", "window partition rank",
      ["Rank products by price within their category.",
       "Show each product's price rank inside its own category.",
       "Produce a price ranking per category."],
      """SELECT category_id, product_name, unit_price,
                RANK() OVER (PARTITION BY category_id ORDER BY unit_price DESC)
                AS price_rank
         FROM products ORDER BY category_id, price_rank"""),

    T("v20", "very_hard", "sales", "window partition row_number join",
      ["Show the top {small_n} customers by revenue in each country.",
       "For every country, list its {small_n} biggest spenders.",
       "Who are the {small_n} highest spending customers per country?"],
      """WITH ranked AS (
             SELECT c.country, c.customer_name, SUM(o.total_amount) AS revenue,
                    ROW_NUMBER() OVER (PARTITION BY c.country
                                       ORDER BY SUM(o.total_amount) DESC) AS rn
             FROM customers c JOIN orders o ON o.customer_id = c.customer_id
             GROUP BY c.country, c.customer_id, c.customer_name)
         SELECT country, customer_name, revenue FROM ranked
         WHERE rn <= {small_n} ORDER BY country, revenue DESC"""),

    T("v21", "very_hard", "finance", "window running_total date",
      ["Show cumulative payments received by month in {year}.",
       "Calculate the running total of collections during {year}.",
       "What is the month-by-month cumulative payment total for {year}?"],
      """SELECT DATE_TRUNC('month', payment_date) AS month,
                SUM(amount) AS monthly_total,
                SUM(SUM(amount)) OVER (
                    ORDER BY DATE_TRUNC('month', payment_date)) AS running_total
         FROM payments WHERE EXTRACT(YEAR FROM payment_date) = {year}
         GROUP BY month ORDER BY month"""),

    T("v22", "very_hard", "sales", "window lag date",
      ["Show year-on-year revenue change.",
       "Compare each year's revenue with the previous year.",
       "What is the annual revenue growth?"],
      """SELECT EXTRACT(YEAR FROM order_date) AS year,
                SUM(total_amount) AS revenue,
                SUM(total_amount) - LAG(SUM(total_amount)) OVER (
                    ORDER BY EXTRACT(YEAR FROM order_date)) AS change_from_previous
         FROM orders GROUP BY year ORDER BY year"""),

    T("v23", "very_hard", "finance", "date_diff avg join",
      ["How many days on average pass between an order and its payment?",
       "Calculate mean days from order to payment.",
       "What is the average collection delay in days?"],
      """SELECT AVG(EXTRACT(EPOCH FROM (p.payment_date - o.order_date)) / 86400.0)
                AS avg_days_to_payment
         FROM payments p JOIN orders o ON o.order_id = p.order_id
         WHERE p.status = 'completed'"""),

    T("v24", "very_hard", "hr", "window partition dense_rank",
      ["Rank employees by salary within their department.",
       "Show each employee's pay rank inside their own department.",
       "Produce a salary ranking per department."],
      """SELECT department_id, first_name, last_name, salary,
                DENSE_RANK() OVER (PARTITION BY department_id ORDER BY salary DESC)
                AS salary_rank
         FROM employees ORDER BY department_id, salary_rank"""),

    T("v25", "very_hard", "hr", "percentile group_by",
      ["What is the median salary in each department?",
       "Show the 50th percentile of pay by department id.",
       "Calculate median salary per department."],
      """SELECT department_id,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
         FROM employees GROUP BY department_id ORDER BY median_salary DESC"""),

    T("v26", "very_hard", "sales", "window running_total date",
      ["Show the cumulative number of customers by signup month.",
       "Calculate customer growth month by month.",
       "How has our customer base accumulated over time?"],
      """SELECT DATE_TRUNC('month', signup_date) AS month,
                COUNT(*) AS new_customers,
                SUM(COUNT(*)) OVER (ORDER BY DATE_TRUNC('month', signup_date))
                AS cumulative_customers
         FROM customers GROUP BY month ORDER BY month"""),

    T("v27", "very_hard", "finance", "conditional_aggregation date",
      ["For each year, show how much was paid by each payment method.",
       "Break down yearly collections by method.",
       "Compare payment methods across years."],
      """SELECT EXTRACT(YEAR FROM payment_date) AS year,
                SUM(amount) FILTER (WHERE payment_method = 'credit_card')
                    AS credit_card,
                SUM(amount) FILTER (WHERE payment_method = 'bank_transfer')
                    AS bank_transfer,
                SUM(amount) FILTER (WHERE payment_method = 'paypal') AS paypal
         FROM payments GROUP BY year ORDER BY year"""),

    T("v28", "very_hard", "logistics", "conditional_aggregation group_by",
      ["For each warehouse, how many shipments were delivered versus lost?",
       "Show delivered and lost shipment counts per warehouse.",
       "Compare delivery outcomes by warehouse."],
      """SELECT w.warehouse_name,
                COUNT(*) FILTER (WHERE s.status = 'delivered') AS delivered,
                COUNT(*) FILTER (WHERE s.status = 'lost') AS lost
         FROM shipments s JOIN warehouses w ON w.warehouse_id = s.warehouse_id
         GROUP BY w.warehouse_id, w.warehouse_name ORDER BY delivered DESC"""),

    T("v29", "very_hard", "sales", "nested_aggregation having",
      ["Which countries have an above-average number of orders per customer?",
       "Show countries where customers order more often than average.",
       "List countries with above-average orders per customer."],
      """WITH per_country AS (
             SELECT c.country, COUNT(o.order_id)::numeric
                    / COUNT(DISTINCT c.customer_id) AS orders_per_customer
             FROM customers c JOIN orders o ON o.customer_id = c.customer_id
             GROUP BY c.country)
         SELECT country, orders_per_customer FROM per_country
         WHERE orders_per_customer > (SELECT AVG(orders_per_customer) FROM per_country)
         ORDER BY orders_per_customer DESC"""),

    T("v30", "very_hard", "catalogue", "window ntile",
      ["Split products into {small_n} price bands.",
       "Divide products into {small_n} equal groups by price.",
       "Assign each product to one of {small_n} price tiers."],
      """SELECT product_name, unit_price,
                NTILE({small_n}) OVER (ORDER BY unit_price DESC) AS price_band
         FROM products ORDER BY unit_price DESC"""),
]

ENTERPRISE_EXTRA: list[Template] = [
    T("x21", "enterprise", "logistics", "business_rule not_exists",
      ["Which paid orders have never been shipped?",
       "Show orders that were paid but never dispatched.",
       "List paid orders still awaiting shipment."],
      """SELECT o.order_id, o.total_amount, o.status FROM orders o
         WHERE EXISTS (SELECT 1 FROM payments p
                       WHERE p.order_id = o.order_id AND p.status = 'completed')
           AND NOT EXISTS (SELECT 1 FROM shipments s WHERE s.order_id = o.order_id)
         ORDER BY o.order_id"""),

    T("x22", "enterprise", "hr", "left_join null_check business_rule",
      ["Which employees have never handled an order?",
       "Show staff with no sales attributed to them.",
       "List employees who have not processed any order."],
      """SELECT e.employee_id, e.first_name, e.last_name
         FROM employees e LEFT JOIN orders o ON o.employee_id = e.employee_id
         WHERE o.order_id IS NULL"""),

    T("x23", "enterprise", "logistics", "left_join null_check",
      ["Which warehouses hold no inventory at all?",
       "Show empty warehouses.",
       "List warehouses with nothing stored in them."],
      """SELECT w.warehouse_id, w.warehouse_name
         FROM warehouses w LEFT JOIN inventory i ON i.warehouse_id = w.warehouse_id
         WHERE i.inventory_id IS NULL"""),

    T("x24", "enterprise", "finance", "null_handling business_rule join having",
      ["Which business customers have outstanding orders exceeding their credit limit?",
       "Show business accounts whose order value is above their credit limit.",
       "List customers over their credit limit."],
      """SELECT c.customer_name, c.credit_limit, SUM(o.total_amount) AS order_value
         FROM customers c JOIN orders o ON o.customer_id = c.customer_id
         WHERE c.credit_limit IS NOT NULL AND o.status NOT IN ('cancelled', 'returned')
         GROUP BY c.customer_id, c.customer_name, c.credit_limit
         HAVING SUM(o.total_amount) > c.credit_limit
         ORDER BY order_value DESC"""),

    T("x25", "enterprise", "catalogue", "business_rule join null_handling",
      ["Which active products come from suppliers rated below {supplier_rating}?",
       "Show products sourced from suppliers rated under {supplier_rating}.",
       "List items supplied by companies scoring under {supplier_rating}."],
      """SELECT p.product_name, s.supplier_name, s.rating
         FROM products p JOIN suppliers s ON s.supplier_id = p.supplier_id
         WHERE p.is_discontinued = FALSE
           AND s.rating IS NOT NULL AND s.rating < {supplier_rating}
         ORDER BY s.rating"""),

    T("x26", "enterprise", "catalogue", "business_rule margin null_handling",
      ["Which products are sold at or below cost?",
       "Show items with zero or negative margin.",
       "List products where the price does not exceed the cost."],
      """SELECT product_id, product_name, unit_price, unit_cost FROM products
         WHERE unit_cost IS NOT NULL AND unit_price <= unit_cost"""),

    T("x27", "enterprise", "finance", "business_rule group_by",
      ["How much money has been refunded, broken down by payment method?",
       "Show refunded amounts per method.",
       "Which payment methods see the most refunds?"],
      """SELECT payment_method, COUNT(*) AS refund_count, SUM(amount) AS refunded
         FROM payments WHERE status = 'refunded'
         GROUP BY payment_method ORDER BY refunded DESC"""),

    T("x28", "enterprise", "cross_domain", "cross_domain date business_rule",
      ["Which countries suffer the most late deliveries?",
       "Show late delivery counts by customer country.",
       "Where are deliveries most often late?"],
      """SELECT c.country, COUNT(*) AS late_deliveries
         FROM shipments s
         JOIN orders o ON o.order_id = s.order_id
         JOIN customers c ON c.customer_id = o.customer_id
         WHERE s.delivered_date IS NOT NULL AND o.required_date IS NOT NULL
           AND s.delivered_date::date > o.required_date
         GROUP BY c.country ORDER BY late_deliveries DESC"""),

    T("x29", "enterprise", "sales", "business_rule date",
      ["Which orders have been pending for more than {pending_days} days?",
       "Show orders stuck in pending status beyond {pending_days} days.",
       "List stale pending orders older than {pending_days} days."],
      """SELECT order_id, customer_id, order_date, total_amount FROM orders
         WHERE status = 'pending'
           AND order_date < {as_of} - INTERVAL '{pending_days} days'
         ORDER BY order_date"""),

    T("x30", "enterprise", "sales", "business_rule data_quality",
      ["Which inactive customers have placed orders in the last "
       "{inactive_recent_months} months?",
       "Show inactive customers who ordered within the past "
       "{inactive_recent_months} months.",
       "Find inactive accounts that have ordered in the last "
       "{inactive_recent_months} months."],
      """SELECT c.customer_id, c.customer_name, MAX(o.order_date) AS last_order
         FROM customers c JOIN orders o ON o.customer_id = c.customer_id
         WHERE c.is_active = FALSE
           AND o.order_date > {as_of} - INTERVAL '{inactive_recent_months} months'
         GROUP BY c.customer_id, c.customer_name ORDER BY last_order DESC"""),

    T("x31", "enterprise", "cross_domain", "cross_domain business_rule join",
      ["Which categories have the most stock sitting below reorder level?",
       "Show understocked categories.",
       "Where are our biggest restocking gaps by category?"],
      """SELECT c.category_name, COUNT(*) AS understocked_items
         FROM inventory i
         JOIN products p ON p.product_id = i.product_id
         JOIN categories c ON c.category_id = p.category_id
         WHERE i.quantity_on_hand < i.reorder_level
         GROUP BY c.category_id, c.category_name ORDER BY understocked_items DESC"""),

    T("x32", "enterprise", "sales", "ambiguous business_rule",
      ["What is our average revenue per customer?",
       "How much is a customer worth on average?",
       "Show mean lifetime value per customer."],
      """SELECT AVG(customer_revenue) AS avg_revenue_per_customer FROM (
             SELECT o.customer_id, SUM(o.total_amount) AS customer_revenue
             FROM orders o WHERE o.status NOT IN ('cancelled', 'returned')
             GROUP BY o.customer_id) t"""),
]


ALL_TEMPLATES: list[Template] = (
    EASY + MEDIUM + HARD + VERY_HARD + ENTERPRISE
    + EASY_EXTRA + MEDIUM_EXTRA + HARD_EXTRA + VERY_HARD_EXTRA + ENTERPRISE_EXTRA
)


def templates_by_difficulty() -> dict[str, list[Template]]:
    grouped: dict[str, list[Template]] = {}
    for template in ALL_TEMPLATES:
        grouped.setdefault(template.difficulty, []).append(template)
    return grouped


def validate_unique_ids(templates: list[Template] | None = None) -> None:
    """Fail fast if two templates share an id — that would corrupt the split."""
    seen: set[str] = set()
    for template in (ALL_TEMPLATES if templates is None else templates):
        if template.template_id in seen:
            raise ValueError(f"duplicate template_id: {template.template_id}")
        seen.add(template.template_id)


def validate_phrasings_cover_sql_slots(templates: list[Template] | None = None) -> None:
    """Every phrasing must mention every slot the SQL depends on.

    If the SQL filters on {days} but a phrasing does not mention it, then all
    values of {days} collapse to the same question text while producing
    different SQL. The result is one question with several conflicting gold
    answers — unanswerable, and it silently corrupts execution-based scoring.
    """
    problems: list[str] = []
    for template in (ALL_TEMPLATES if templates is None else templates):
        sql_slots = set(SLOT_PATTERN.findall(template.sql))
        for i, question in enumerate(template.questions):
            missing = sql_slots - set(SLOT_PATTERN.findall(question))
            if missing:
                problems.append(
                    f"{template.template_id} phrasing[{i}] omits "
                    f"{sorted(missing)}: {question!r}"
                )
    if problems:
        raise ValueError(
            "phrasings missing SQL slots:\n  " + "\n  ".join(problems)
        )


def validate_no_question_collisions(templates: list[Template] | None = None) -> None:
    """Two templates must not share a phrasing unless their SQL is identical."""
    by_question: dict[str, list[Template]] = {}
    for template in (ALL_TEMPLATES if templates is None else templates):
        for question in template.questions:
            by_question.setdefault(question, []).append(template)

    problems = [
        f"{question!r} used by {sorted(t.template_id for t in templates)}"
        for question, templates in by_question.items()
        if len(templates) > 1 and len({t.sql for t in templates}) > 1
    ]
    if problems:
        raise ValueError(
            "identical questions mapping to different SQL:\n  "
            + "\n  ".join(problems)
        )


def validate_all(templates: list[Template] | None = None) -> None:
    """Run every structural check. ``templates`` defaults to the v1 list;
    ``templates_v2`` passes its own so the checks are shared, not copied."""
    validate_unique_ids(templates)
    validate_phrasings_cover_sql_slots(templates)
    validate_no_question_collisions(templates)
