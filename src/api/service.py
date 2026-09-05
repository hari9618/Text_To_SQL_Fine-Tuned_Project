"""The Text-to-SQL pipeline, assembled.

Phase 12.

This is the architecture from CLAUDE.md, in one class:

    question -> schema context -> model -> SQL
             -> static validation -> PostgreSQL
             -> success                        -> rows
             -> failure -> repair -> PostgreSQL -> rows

Nothing here is new logic. Generation, validation, execution and repair are the
same modules the benchmark used, which is the point: the number the API
delivers in production is the number Phase 10 and Phase 11 measured. A service
that reimplemented any of these would be measuring one system and shipping
another.

Safety comes from three layers that do not depend on the model behaving:

* the database role is not a superuser (``app_config``),
* every session is read-only with a statement timeout (``read_only_connection``),
* the SQL is statically rejected unless it is a single read-only statement over
  tables that exist (``validator``).

The model is never trusted. It is only asked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import psycopg

from src.model.base_model import TextToSQLModel, extract_sql
from src.model.prompt import build_messages, prompt_fingerprint  # noqa: F401
from src.model.repair_prompt import build_repair_messages
from src.model.schema_context import build_schema_context
from src.sql.executor import SchemaInfo, load_schema_info
from src.sql.repair import diagnose

# Server-side ceiling. A caller may ask for fewer rows, never more: the point of
# a cap is that it does not depend on the caller being reasonable.
HARD_ROW_CAP = 1000


@dataclass
class PipelineResult:
    """Everything one question produced."""

    ok: bool
    sql: str | None
    columns: list[str]
    rows: list[list[Any]]
    row_count: int | None
    truncated: bool
    repaired: bool
    attempts: list[dict[str, Any]]
    error: str | None
    error_stage: str | None
    generation_ms: float
    repair_generation_ms: float
    execution_ms: float
    total_ms: float


class TextToSQLService:
    """Holds the schema context and the model; answers questions.

    The schema is rendered once at start-up rather than per request. It is
    ~5.5 kB of text read from the catalog, identical for every question, and
    re-reading it would add a round trip to the hot path for no benefit. A
    schema change requires a restart, which is the correct trade for a service
    whose accuracy figures are tied to a specific schema fingerprint.
    """

    def __init__(
        self,
        model: TextToSQLModel,
        schema_text: str,
        schema_fingerprint: str,
        schema_info: SchemaInfo,
    ) -> None:
        self.model = model
        self.schema_text = schema_text
        self.schema_fingerprint = schema_fingerprint
        self.schema_info = schema_info

    @classmethod
    def from_connection(
        cls, model: TextToSQLModel, conn: psycopg.Connection
    ) -> "TextToSQLService":
        schema_text, fingerprint = build_schema_context(conn)
        return cls(model, schema_text, fingerprint, load_schema_info(conn))

    # -- generation ---------------------------------------------------------

    def _generate(self, question: str) -> tuple[str, float, str | None]:
        """Ask the model for SQL. Returns (sql, latency_ms, error)."""
        gen = self.model.generate(question, self.schema_text)
        if not gen.ok:
            return "", gen.latency_ms, gen.error or "generation failed"
        return gen.sql, gen.latency_ms, None

    def _regenerate(self, question: str, failed_sql: str,
                    error: str) -> tuple[str, float, str | None]:
        """Ask the model to fix SQL, using the repair prompt.

        ``TextToSQLModel.generate`` takes (question, schema), so the repair
        prompt is rendered here and passed through the same interface. Models
        that accept raw messages can override ``generate_messages``; the
        fallback keeps every backend working without special-casing.
        """
        messages = build_repair_messages(
            question, self.schema_text, failed_sql, error)
        generate_messages = getattr(self.model, "generate_messages", None)
        started = time.perf_counter()
        if callable(generate_messages):
            gen = generate_messages(messages)
        else:
            # The repair prompt's user turn already contains the schema, the
            # failing query and the error, so it is self-contained.
            gen = self.model.generate(messages[1]["content"], "")
        elapsed = (time.perf_counter() - started) * 1000
        if not gen.ok:
            return "", elapsed, gen.error or "repair generation failed"
        return gen.sql, getattr(gen, "latency_ms", elapsed) or elapsed, None

    # -- the pipeline -------------------------------------------------------

    def answer(
        self,
        question: str,
        conn: psycopg.Connection,
        max_rows: int = 100,
        repair: bool = True,
        include_rows: bool = True,
    ) -> PipelineResult:
        """Run one question end to end. Never raises on bad SQL."""
        started = time.perf_counter()
        limit = max(1, min(int(max_rows), HARD_ROW_CAP))
        attempts: list[dict[str, Any]] = []

        sql, generation_ms, gen_error = self._generate(question)
        repair_ms = 0.0
        execution_ms = 0.0

        if gen_error:
            return PipelineResult(
                ok=False, sql=None, columns=[], rows=[], row_count=None,
                truncated=False, repaired=False,
                attempts=[{"sql": "", "ok": False, "stage": "generation",
                           "error": gen_error}],
                error=gen_error, error_stage="generation",
                generation_ms=generation_ms, repair_generation_ms=0.0,
                execution_ms=0.0,
                total_ms=(time.perf_counter() - started) * 1000,
            )

        diagnosis = diagnose(sql, conn, self.schema_info)
        execution_ms += diagnosis.execution.latency_ms if diagnosis.execution else 0.0
        attempts.append({"sql": sql, "ok": diagnosis.ok,
                         "stage": diagnosis.stage, "error": diagnosis.error})

        repaired = False
        if not diagnosis.ok and repair:
            fixed_sql, repair_ms, repair_error = self._regenerate(
                question, sql, diagnosis.error or "unknown error")
            if repair_error:
                attempts.append({"sql": "", "ok": False, "stage": "generation",
                                 "error": repair_error})
            elif fixed_sql:
                repaired = True
                sql = fixed_sql
                diagnosis = diagnose(sql, conn, self.schema_info)
                execution_ms += (diagnosis.execution.latency_ms
                                 if diagnosis.execution else 0.0)
                attempts.append({"sql": sql, "ok": diagnosis.ok,
                                 "stage": diagnosis.stage,
                                 "error": diagnosis.error})

        columns: list[str] = []
        rows: list[list[Any]] = []
        row_count: int | None = None
        truncated = False

        if diagnosis.ok:
            row_count = diagnosis.execution.row_count if diagnosis.execution else 0
            columns = list(diagnosis.execution.columns) if diagnosis.execution else []
            if include_rows:
                rows, truncated = self._fetch_rows(conn, sql, limit)
                # `truncated` from the fetch means more rows matched than the
                # caller asked for; the executor's own flag means more matched
                # than the harness ceiling. Either is worth reporting.
                truncated = truncated or bool(
                    diagnosis.execution and diagnosis.execution.truncated)

        return PipelineResult(
            ok=diagnosis.ok, sql=sql, columns=columns, rows=rows,
            row_count=row_count, truncated=truncated, repaired=repaired,
            attempts=attempts,
            error=None if diagnosis.ok else diagnosis.error,
            error_stage=None if diagnosis.ok else diagnosis.stage,
            generation_ms=round(generation_ms, 2),
            repair_generation_ms=round(repair_ms, 2),
            execution_ms=round(execution_ms, 2),
            total_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    @staticmethod
    def _fetch_rows(
        conn: psycopg.Connection, sql: str, limit: int
    ) -> tuple[list[list[Any]], bool]:
        """Re-run the validated query and pull at most ``limit`` rows.

        A second execution rather than one that keeps its cursor open: the
        diagnosis step already proved the statement is a safe, working SELECT,
        and holding a cursor across the repair decision would keep a
        transaction open for the length of a model call.

        Values are stringified. JSON has no representation for Decimal, date or
        UUID, and silently coercing them to float would corrupt monetary
        amounts — which is exactly the kind of quiet wrongness this project is
        about avoiding.
        """
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                if cur.description is None:
                    return [], False
                fetched = cur.fetchmany(limit + 1)
        except psycopg.Error:
            return [], False

        truncated = len(fetched) > limit
        return [
            [None if v is None else str(v) for v in row]
            for row in fetched[:limit]
        ], truncated
