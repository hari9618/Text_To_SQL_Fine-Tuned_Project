"""API and pipeline tests.

Phase 12.

These run against the real database with a **stub model**, which is the point:
the model is the one part of the system that cannot be tested deterministically,
so it is replaced with canned SQL and everything around it — validation,
execution, the repair loop, row caps, the HTTP contract, error handling — is
tested for real.

A test suite that mocked the database instead would prove nothing: the entire
safety argument rests on PostgreSQL actually refusing to write and actually
timing out.

Run from the project root:

    env\\Scripts\\python.exe -m pytest tests/test_api.py -v
"""

from __future__ import annotations

import os
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.backends import StubModel, build_model
from src.api.schemas import MAX_QUESTION_CHARS
from src.api.service import HARD_ROW_CAP, TextToSQLService
from src.sql.config import app_config
from src.sql.executor import load_schema_info, read_only_connection

# Real SQL against the real schema, so a passing test means the query genuinely
# ran. Kept trivial on purpose: this suite tests the pipeline, not the model.
GOOD_SQL = "SELECT customer_id, customer_name FROM customers ORDER BY customer_id"
UNKNOWN_COLUMN_SQL = "SELECT nonexistent_column FROM customers"
UNKNOWN_TABLE_SQL = "SELECT * FROM not_a_real_table"
UNPARSEABLE_SQL = "SELECT FROM WHERE ORDER"
WRITE_SQL = "DROP TABLE customers"


@pytest.fixture(scope="module")
def db():
    with read_only_connection(app_config()) as conn:
        yield conn


@pytest.fixture(scope="module")
def schema_info(db):
    return load_schema_info(db)


def make_service(model: StubModel, db, schema_info) -> TextToSQLService:
    return TextToSQLService(model, "SCHEMA TEXT", "d03619e711661bc5", schema_info)


# ---------------------------------------------------------------- pipeline --

def test_successful_query_returns_rows(db, schema_info):
    svc = make_service(StubModel(default=GOOD_SQL), db, schema_info)
    r = svc.answer("List customers", db, max_rows=5)

    assert r.ok
    assert r.sql == GOOD_SQL
    assert r.columns == ["customer_id", "customer_name"]
    assert len(r.rows) == 5
    assert r.truncated is True          # more customers exist than 5
    assert r.repaired is False
    assert r.execution_ms > 0


def test_row_cap_is_enforced_server_side(db, schema_info):
    """A caller asking for more than the ceiling gets the ceiling."""
    svc = make_service(StubModel(default=GOOD_SQL), db, schema_info)
    r = svc.answer("List customers", db, max_rows=10_000_000)
    assert len(r.rows) <= HARD_ROW_CAP


def test_include_rows_false_still_reports_shape(db, schema_info):
    svc = make_service(StubModel(default=GOOD_SQL), db, schema_info)
    r = svc.answer("List customers", db, include_rows=False)
    assert r.ok
    assert r.rows == []
    assert r.row_count and r.row_count > 0


def test_write_statement_is_rejected_before_execution(db, schema_info):
    """The validator must stop this; the read-only session is the second line."""
    svc = make_service(StubModel(default=WRITE_SQL), db, schema_info)
    r = svc.answer("Delete everything", db, repair=False)

    assert r.ok is False
    assert r.error_stage == "static"
    assert "SELECT" in (r.error or "")
    # The table must still be there.
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM customers")
        assert cur.fetchone()[0] > 0


def test_unknown_table_is_rejected_statically(db, schema_info):
    svc = make_service(StubModel(default=UNKNOWN_TABLE_SQL), db, schema_info)
    r = svc.answer("Query a fake table", db, repair=False)
    assert r.ok is False
    assert r.error_stage == "static"
    assert "not_a_real_table" in (r.error or "")


def test_unparseable_sql_is_rejected_statically(db, schema_info):
    svc = make_service(StubModel(default=UNPARSEABLE_SQL), db, schema_info)
    r = svc.answer("Nonsense", db, repair=False)
    assert r.ok is False
    assert r.error_stage == "static"


def test_unknown_column_fails_at_execution(db, schema_info):
    """sqlglot checks tables, PostgreSQL checks columns."""
    svc = make_service(StubModel(default=UNKNOWN_COLUMN_SQL), db, schema_info)
    r = svc.answer("Bad column", db, repair=False)
    assert r.ok is False
    assert r.error_stage == "execution"
    assert "nonexistent_column" in (r.error or "")


# ------------------------------------------------------------------ repair --

def test_repair_fixes_a_failing_query(db, schema_info):
    model = StubModel(
        default=UNKNOWN_COLUMN_SQL,
        repair_responses={"nonexistent_column": GOOD_SQL},
    )
    svc = make_service(model, db, schema_info)
    r = svc.answer("Bad column", db, max_rows=3, repair=True)

    assert r.ok is True
    assert r.repaired is True
    assert r.sql == GOOD_SQL
    assert len(r.rows) == 3
    assert len(r.attempts) == 2
    assert r.attempts[0]["ok"] is False
    assert r.attempts[1]["ok"] is True


def test_repair_prompt_carries_the_error_and_no_gold(db, schema_info):
    """The repair call must see the error -- and must not see an answer."""
    model = StubModel(default=UNKNOWN_COLUMN_SQL,
                      repair_responses={"nonexistent_column": GOOD_SQL})
    svc = make_service(model, db, schema_info)
    svc.answer("Bad column", db, repair=True)

    assert len(model.repair_calls) == 1
    content = model.repair_calls[0][-1]["content"]
    assert "nonexistent_column" in content       # the failing SQL
    assert "does not exist" in content           # the database's error
    assert UNKNOWN_COLUMN_SQL in content
    assert GOOD_SQL not in content               # no correct answer leaked


def test_repair_disabled_makes_no_second_call(db, schema_info):
    model = StubModel(default=UNKNOWN_COLUMN_SQL,
                      repair_responses={"nonexistent_column": GOOD_SQL})
    svc = make_service(model, db, schema_info)
    r = svc.answer("Bad column", db, repair=False)

    assert r.ok is False
    assert r.repaired is False
    assert model.repair_calls == []
    assert len(r.attempts) == 1


def test_repair_that_also_fails_is_reported_honestly(db, schema_info):
    model = StubModel(default=UNKNOWN_COLUMN_SQL,
                      repair_responses={"nonexistent_column": UNKNOWN_TABLE_SQL})
    svc = make_service(model, db, schema_info)
    r = svc.answer("Bad column", db, repair=True)

    assert r.ok is False
    assert r.repaired is True          # an attempt was made
    assert len(r.attempts) == 2
    assert all(not a["ok"] for a in r.attempts)


def test_only_one_repair_attempt(db, schema_info):
    """Unbounded retries would turn one bad question into a latency spike."""
    model = StubModel(default=UNKNOWN_COLUMN_SQL,
                      repair_responses={"nonexistent_column": UNKNOWN_COLUMN_SQL})
    svc = make_service(model, db, schema_info)
    r = svc.answer("Bad column", db, repair=True)
    assert len(model.repair_calls) == 1
    assert len(r.attempts) == 2


# -------------------------------------------------------------------- HTTP --

@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    os.environ["MODEL_BACKEND"] = "stub"
    from src.api.main import app

    with TestClient(app) as c:
        yield c


def test_health_reports_database_and_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["database"] is True
    assert body["model"] is True
    assert body["status"] == "ok"
    assert body["prompt_fingerprint"] == "8288e41a496531a9"


def test_schema_endpoint_matches_the_frozen_fingerprint(client):
    r = client.get("/schema")
    assert r.status_code == 200
    body = r.json()
    assert body["fingerprint"] == "d03619e711661bc5"
    assert "customers" in body["tables"]


def test_query_endpoint_returns_200_and_rows(client):
    r = client.post("/query", json={"question": "anything", "max_rows": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["sql"]
    assert body["timings"]["total_ms"] >= 0


def test_failing_sql_is_200_not_500(client):
    """A model mistake is not a server fault; 5xx must stay meaningful."""
    from src.api.main import app

    app.state.service.model = StubModel(default=UNKNOWN_COLUMN_SQL)
    try:
        r = client.post("/query", json={"question": "bad", "repair": False})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        assert body["error_stage"] == "execution"
    finally:
        app.state.service.model = StubModel(default=GOOD_SQL)


def test_question_length_is_validated(client):
    r = client.post("/query", json={"question": "x" * (MAX_QUESTION_CHARS + 1)})
    assert r.status_code == 422


def test_empty_question_is_rejected(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_max_rows_out_of_range_is_rejected(client):
    assert client.post(
        "/query", json={"question": "x", "max_rows": 0}).status_code == 422


def test_no_credentials_leak_in_any_response(client):
    """Nothing the service returns may contain the database password."""
    secret = app_config().password
    bodies = [
        client.get("/health").text,
        client.get("/schema").text,
        client.post("/query", json={"question": "anything"}).text,
        client.get("/").text,
    ]
    for body in bodies:
        assert secret not in body
        assert "password" not in body.lower()


def test_unknown_backend_fails_loudly():
    with pytest.raises(RuntimeError, match="unknown MODEL_BACKEND"):
        build_model("does-not-exist")


# --------------------------------------------------------------- demo UI --

def test_root_serves_the_demo_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "<title>Enterprise Text-to-SQL</title>" in body
    # The UI drives the same public endpoints, not a private one.
    for endpoint in ("/query", "/health", "/schema"):
        assert endpoint in body


def test_ui_contains_no_credentials(client):
    """The page is served to anyone who can reach the service."""
    body = client.get("/").text
    assert app_config().password not in body
    assert app_config().user not in body


def test_ui_is_self_contained(client):
    """No CDN, no build step: the container ships one file that just works."""
    body = client.get("/").text
    for pattern in ("http://", "https://cdn", "<script src=", "<link rel=\"stylesheet\""):
        assert pattern not in body, f"UI reaches outside the container: {pattern}"
