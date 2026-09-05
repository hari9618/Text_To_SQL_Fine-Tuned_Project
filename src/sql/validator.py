"""Static SQL checks: parsing, statement classification, schema grounding.

Phase 3 module. Phase 11 (SQL repair) and Phase 12 (the API) reuse it to screen
model-generated SQL before it reaches the database.

These checks run *before* execution and answer different questions than
execution does:

- Execution tells you whether PostgreSQL accepted the query. It is the
  authority on whether a table or column exists.
- Static analysis tells you what a query *is* — whether it reads or writes,
  whether it is one statement or several — which you want to know before
  handing it to a database at all.

Both are needed. A ``DROP TABLE customers`` executes perfectly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp

DIALECT = "postgres"

# Node types that only read. Anything else is treated as a write or an
# administrative command and rejected.
READ_ONLY_NODES = (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.Subquery)


@dataclass
class StaticCheck:
    """Result of analysing one SQL string without executing it."""

    parseable: bool
    is_read_only: bool
    statement_count: int
    referenced_tables: tuple[str, ...] = ()
    unknown_tables: tuple[str, ...] = ()
    parse_error: str | None = None
    statement_kind: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.parseable
            and self.is_read_only
            and self.statement_count == 1
            and not self.unknown_tables
        )

    def failure_reason(self) -> str | None:
        if not self.parseable:
            return "unparseable_sql"
        if self.statement_count != 1:
            return "multiple_statements"
        if not self.is_read_only:
            return "dangerous_sql"
        if self.unknown_tables:
            return "unknown_table"
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "parseable": self.parseable,
            "is_read_only": self.is_read_only,
            "statement_count": self.statement_count,
            "referenced_tables": list(self.referenced_tables),
            "unknown_tables": list(self.unknown_tables),
            "parse_error": self.parse_error,
            "statement_kind": self.statement_kind,
        }


def _local_names(expression: exp.Expression) -> set[str]:
    """Names that look like tables but are defined inside the query itself.

    CTE names and derived-table aliases appear as table references in the AST.
    A recursive CTE even references itself. None of them are real tables, so
    they must be excluded before checking names against the catalog.
    """
    names: set[str] = set()
    for cte in expression.find_all(exp.CTE):
        if cte.alias_or_name:
            names.add(cte.alias_or_name.lower())
    for subquery in expression.find_all(exp.Subquery):
        if subquery.alias:
            names.add(str(subquery.alias).lower())
    return names


def analyse(sql: str, known_tables: set[str] | None = None) -> StaticCheck:
    """Parse one SQL string and describe it. Never raises."""
    try:
        statements = [s for s in sqlglot.parse(sql, dialect=DIALECT) if s is not None]
    except Exception as exc:  # sqlglot raises several parse error types
        return StaticCheck(
            parseable=False,
            is_read_only=False,
            statement_count=0,
            parse_error=str(exc).strip().splitlines()[0][:300],
        )

    if not statements:
        return StaticCheck(
            parseable=False,
            is_read_only=False,
            statement_count=0,
            parse_error="no statement found",
        )

    kinds = [type(s).__name__ for s in statements]
    read_only = all(isinstance(s, READ_ONLY_NODES) for s in statements)

    referenced: set[str] = set()
    local: set[str] = set()
    for statement in statements:
        local |= _local_names(statement)
        for table in statement.find_all(exp.Table):
            if table.name:
                referenced.add(table.name.lower())

    real = sorted(referenced - local)
    unknown: list[str] = []
    if known_tables is not None:
        known_lower = {t.lower() for t in known_tables}
        unknown = sorted(t for t in real if t not in known_lower)

    return StaticCheck(
        parseable=True,
        is_read_only=read_only,
        statement_count=len(statements),
        referenced_tables=tuple(real),
        unknown_tables=tuple(unknown),
        statement_kind=",".join(sorted(set(kinds))),
    )


@dataclass
class SlotCheck:
    """Whether every templated value survives into the question text."""

    ok: bool
    missing: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "missing": list(self.missing)}


def check_slot_representation(
    question: str, sql: str, slot_values: dict[str, str]
) -> SlotCheck:
    """Every slot value used in the SQL must appear in the question.

    If the SQL filters on 90 days but the question never says "90", the question
    is unanswerable — no reader, human or model, could know which threshold was
    meant. The gold SQL would then be one arbitrary choice among many, and the
    model would be marked wrong for a reasonable answer.

    Values absent from the SQL are ignored: a slot can legitimately appear only
    in the phrasing.
    """
    missing = [
        f"{slot}={value}"
        for slot, value in slot_values.items()
        if str(value) in sql and str(value) not in question
    ]
    return SlotCheck(ok=not missing, missing=tuple(missing))
