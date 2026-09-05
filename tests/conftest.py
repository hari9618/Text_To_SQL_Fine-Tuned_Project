"""Shared pytest fixtures for database tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator

import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sql.config import app_config  # noqa: E402


@pytest.fixture(scope="session")
def conn() -> Iterator[psycopg.Connection]:
    """One read-only connection shared by the whole test session.

    Read-only by default: these tests inspect data, they must never modify it.
    The one test that needs to write opens its own connection.
    """
    with psycopg.connect(app_config().conninfo(), connect_timeout=10) as c:
        c.read_only = True
        yield c


@pytest.fixture(scope="session")
def query(conn: psycopg.Connection):
    """Return a helper that runs SQL and returns all rows."""

    def _query(sql: str, params: tuple | None = None) -> list[tuple[Any, ...]]:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    return _query


@pytest.fixture(scope="session")
def scalar(conn: psycopg.Connection):
    """Return a helper that runs SQL and returns a single value."""

    def _scalar(sql: str, params: tuple | None = None) -> Any:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return row[0] if row else None

    return _scalar
