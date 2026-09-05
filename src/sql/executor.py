"""Execute SQL against PostgreSQL safely and record what happened.

Phase 1 module, first used in earnest by Phase 3 validation. Phase 4 evaluation
and Phase 11 repair reuse it, which is why it lives in src/ rather than in a
script.

Two things make this safe enough to run model-generated SQL through later:

*Read-only sessions.* ``default_transaction_read_only`` is set on the
connection, so even a query that slipped past validation cannot modify data.

*Statement timeouts.* A malformed join across 154k order_items can run for
hours. The timeout turns that into a recorded failure instead of a hang.
"""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import psycopg

from src.sql.config import DatabaseConfig

# PostgreSQL SQLSTATE codes worth distinguishing. Everything else is lumped
# into "other" — the raw message is kept regardless.
ERROR_CLASSES: dict[str, str] = {
    "42601": "syntax_error",
    "42P01": "undefined_table",
    "42703": "undefined_column",
    "42883": "undefined_function",
    "42P18": "indeterminate_datatype",
    "42804": "datatype_mismatch",
    "42702": "ambiguous_column",
    "42P09": "ambiguous_alias",
    "22012": "division_by_zero",
    "57014": "timeout",
}

DEFAULT_STATEMENT_TIMEOUT_MS = 30_000
# Ceiling on rows pulled into memory. No benchmark query should return more;
# if one does, that is itself worth knowing about.
MAX_ROWS = 100_000


@dataclass
class ExecutionResult:
    """Outcome of running one query."""

    ok: bool
    latency_ms: float
    row_count: int | None = None
    columns: tuple[str, ...] = ()
    fingerprint: str | None = None
    # Order-sensitive variant. Needed because "top 10 customers by revenue"
    # has row order as part of its answer, and the headline fingerprint
    # deliberately ignores order.
    ordered_fingerprint: str | None = None
    error_class: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "latency_ms": round(self.latency_ms, 2),
            "row_count": self.row_count,
            "columns": list(self.columns),
            "fingerprint": self.fingerprint,
            "ordered_fingerprint": self.ordered_fingerprint,
            "error_class": self.error_class,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "truncated": self.truncated,
        }


def result_fingerprint(rows: Sequence[Sequence[Any]]) -> str:
    """Order-insensitive hash of a result set.

    Two SQL queries that answer the same question may return their rows in
    different orders, so the fingerprint sorts before hashing. That matches the
    evaluation rule in CLAUDE.md: SELECT COUNT(*) and SELECT COUNT(id) are
    different strings but the same answer.

    Values are stringified rather than compared by type, because equal numbers
    can arrive as Decimal or int depending on the query shape.
    """
    normalised = sorted(
        tuple(("NULL" if v is None else str(v)) for v in row) for row in rows
    )
    digest = hashlib.md5(repr(normalised).encode("utf-8"))
    return digest.hexdigest()


def ordered_result_fingerprint(rows: Sequence[Sequence[Any]]) -> str:
    """Order-sensitive hash of a result set.

    Used for questions where row order is part of the answer — "the top 10
    customers by revenue" is wrong if the ten are returned in a different
    order. Kept separate from the headline fingerprint so that ordering is
    scored only where the gold query actually asked for it.
    """
    normalised = [
        tuple(("NULL" if v is None else str(v)) for v in row) for row in rows
    ]
    return hashlib.md5(repr(normalised).encode("utf-8")).hexdigest()


@contextmanager
def read_only_connection(
    cfg: DatabaseConfig, statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS
) -> Iterator[psycopg.Connection]:
    """Open an autocommit, read-only connection with a statement timeout.

    Autocommit matters here: without it a failed statement leaves the
    transaction in an aborted state and every subsequent query fails with
    "current transaction is aborted", which would look like thousands of
    failures caused by the first one.
    """
    with psycopg.connect(cfg.conninfo(), autocommit=True, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {int(statement_timeout_ms)}")
            cur.execute("SET default_transaction_read_only = on")
        yield conn


def execute(conn: psycopg.Connection, sql: str) -> ExecutionResult:
    """Run one query and describe the outcome. Never raises on SQL errors."""
    started = time.perf_counter()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)

            # A statement with no result set (should not occur for SELECTs).
            if cur.description is None:
                elapsed = (time.perf_counter() - started) * 1000
                return ExecutionResult(ok=True, latency_ms=elapsed, row_count=0)

            columns = tuple(d.name for d in cur.description)
            rows = cur.fetchmany(MAX_ROWS)
            truncated = len(rows) == MAX_ROWS

        elapsed = (time.perf_counter() - started) * 1000
        return ExecutionResult(
            ok=True,
            latency_ms=elapsed,
            row_count=len(rows),
            columns=columns,
            fingerprint=result_fingerprint(rows),
            ordered_fingerprint=ordered_result_fingerprint(rows),
            truncated=truncated,
        )

    except psycopg.Error as exc:
        elapsed = (time.perf_counter() - started) * 1000
        code = getattr(exc, "sqlstate", None)
        return ExecutionResult(
            ok=False,
            latency_ms=elapsed,
            error_class=ERROR_CLASSES.get(code or "", "other"),
            error_code=code,
            error_message=str(exc).strip().splitlines()[0][:300],
        )


@dataclass
class SchemaInfo:
    """Tables and columns that actually exist, for grounding checks."""

    tables: set[str] = field(default_factory=set)
    columns_by_table: dict[str, set[str]] = field(default_factory=dict)

    @property
    def all_columns(self) -> set[str]:
        return {c for cols in self.columns_by_table.values() for c in cols}


def load_schema_info(conn: psycopg.Connection) -> SchemaInfo:
    """Read the real table and column names from the catalog."""
    info = SchemaInfo()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
            """
        )
        for table, column in cur.fetchall():
            info.tables.add(table)
            info.columns_by_table.setdefault(table, set()).add(column)
    return info
