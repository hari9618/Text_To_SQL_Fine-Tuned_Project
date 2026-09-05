"""SQL error detection and repair.

Phase 11.

When generated SQL fails, the failure is usually *legible*: PostgreSQL says
exactly which column does not exist or which reference is ambiguous, and
sqlglot says exactly why a statement will not parse. Handing that message back
to the model along with the question, the schema and its own broken query is
often enough to fix it.

    generate -> validate -> execute
                    |          |
                    +----------+---> failure + error message
                                          |
                                     build repair prompt
                                          |
                                     model regenerates
                                          |
                                     validate -> execute -> result

**Only detectable failures are repaired.** A query that runs cleanly and
returns the wrong rows looks identical to a correct one without the gold
answer, so it is left alone. Repairing on gold would be cheating; the whole
point is that this loop must work in production, where no gold exists.

The repair prompt is *not* what the model was fine-tuned on -- Phase 9 trained
on question-to-SQL only. Repair therefore exercises the fine-tuned model
zero-shot on a task it never saw, which is worth remembering when reading the
Phase 11 numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import psycopg

from src.model.repair_prompt import (  # re-exported: one definition, two callers
    REPAIR_PROMPT_VERSION,
    build_repair_messages,
    repair_prompt_fingerprint,
)
from src.sql import validator
from src.sql.executor import ExecutionResult, SchemaInfo, execute

__all__ = ["REPAIR_PROMPT_VERSION", "build_repair_messages",
           "repair_prompt_fingerprint", "Diagnosis", "diagnose",
           "RepairAttempt", "RepairResult", "repair_loop",
           "describe_static_failure"]

# --------------------------------------------------------------------------
# Failure detection
# --------------------------------------------------------------------------

@dataclass
class Diagnosis:
    """Why a query is unusable, in words a model can act on."""

    ok: bool
    error: str | None = None
    stage: str | None = None          # "static" | "execution"
    execution: ExecutionResult | None = None

    @property
    def repairable(self) -> bool:
        """Detectable without gold. A wrong-but-running query is not."""
        return not self.ok


def describe_static_failure(check: validator.StaticCheck) -> str:
    """Turn a static check into one actionable line."""
    if not check.parseable:
        return f"the query does not parse: {check.parse_error}"
    if check.statement_count != 1:
        return (f"expected a single statement, found {check.statement_count}")
    if not check.is_read_only:
        return (f"only SELECT is allowed, this is {check.statement_kind}")
    if check.unknown_tables:
        return ("these tables do not exist in the schema: "
                + ", ".join(check.unknown_tables))
    return "the query failed validation"


def diagnose(
    sql: str,
    conn: psycopg.Connection,
    schema_info: SchemaInfo,
) -> Diagnosis:
    """Static-check then execute one query, describing the first failure.

    Mirrors the benchmark's own order so a repair triggers on exactly the
    failures the evaluation counts as non-executable.
    """
    if not sql.strip():
        return Diagnosis(ok=False, stage="static",
                         error="no SQL statement was produced")

    check = validator.analyse(sql, known_tables=set(schema_info.tables))
    if not check.ok:
        return Diagnosis(ok=False, stage="static",
                         error=describe_static_failure(check))

    result = execute(conn, sql)
    if not result.ok:
        return Diagnosis(ok=False, stage="execution", execution=result,
                         error=result.error_message or "execution failed")

    return Diagnosis(ok=True, execution=result)


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

@dataclass
class RepairAttempt:
    """One pass through the repair loop."""

    sql: str
    error: str | None
    stage: str | None
    ok: bool
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"sql": self.sql, "error": self.error, "stage": self.stage,
                "ok": self.ok, "latency_ms": round(self.latency_ms, 2)}


@dataclass
class RepairResult:
    """Outcome of generation plus zero or more repairs."""

    sql: str
    ok: bool
    attempts: list[RepairAttempt] = field(default_factory=list)
    repaired: bool = False
    execution: ExecutionResult | None = None

    @property
    def n_repairs(self) -> int:
        return max(0, len(self.attempts) - 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sql": self.sql, "ok": self.ok, "repaired": self.repaired,
            "n_repairs": self.n_repairs,
            "attempts": [a.as_dict() for a in self.attempts],
        }


def repair_loop(
    initial_sql: str,
    question: str,
    schema: str,
    conn: psycopg.Connection,
    schema_info: SchemaInfo,
    regenerate: Callable[[str, str], str],
    max_repairs: int = 1,
) -> RepairResult:
    """Execute, and on a detectable failure ask ``regenerate`` for a fix.

    ``regenerate(failed_sql, error)`` returns corrected SQL. Keeping it a
    callable rather than a model keeps this function testable without a GPU or
    a network call, and lets Phase 12 pass whichever model it is serving.

    ``max_repairs=1`` by default: the second repair on the same query almost
    always repeats the first, and a loop that retries indefinitely turns one
    bad question into an unbounded latency spike in production.
    """
    sql = initial_sql
    attempts: list[RepairAttempt] = []
    execution: ExecutionResult | None = None

    for round_index in range(max_repairs + 1):
        diagnosis = diagnose(sql, conn, schema_info)
        execution = diagnosis.execution
        attempts.append(RepairAttempt(
            sql=sql, error=diagnosis.error, stage=diagnosis.stage,
            ok=diagnosis.ok,
        ))
        if diagnosis.ok:
            return RepairResult(sql=sql, ok=True, attempts=attempts,
                                repaired=round_index > 0, execution=execution)
        if round_index == max_repairs:
            break
        sql = regenerate(sql, diagnosis.error or "unknown error")

    return RepairResult(sql=sql, ok=False, attempts=attempts,
                        repaired=len(attempts) > 1, execution=execution)
