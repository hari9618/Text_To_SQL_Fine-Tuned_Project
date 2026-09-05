"""Request and response models for the Text-to-SQL API.

Phase 12.

The response is deliberately verbose. A service that returns only rows is
impossible to debug and impossible to trust: callers need to see the SQL that
ran, whether it was repaired, and where the time went. Everything here is
information the caller is entitled to; nothing here exposes credentials, the
connection string, or the server's filesystem.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

MAX_QUESTION_CHARS = 2000
DEFAULT_MAX_ROWS = 100
ROW_CEILING = 1000


class QueryRequest(BaseModel):
    """One natural-language question."""

    question: str = Field(
        ..., min_length=1, max_length=MAX_QUESTION_CHARS,
        description="The business question, in plain English.",
        examples=["Which five customers spent the most last year?"],
    )
    max_rows: int = Field(
        DEFAULT_MAX_ROWS, ge=1, le=ROW_CEILING,
        description="Ceiling on rows returned. The server caps this regardless.",
    )
    repair: bool = Field(
        True,
        description="Retry once with the database error fed back if the "
                    "generated SQL fails.",
    )
    include_rows: bool = Field(
        True,
        description="Set false to validate and time a query without "
                    "transferring its result set.",
    )


class AttemptInfo(BaseModel):
    """One pass through generate-validate-execute."""

    sql: str
    ok: bool
    stage: str | None = Field(
        None, description="Where it failed: 'static' or 'execution'.")
    error: str | None = None


class Timings(BaseModel):
    """Milliseconds, split so a slow request can be attributed."""

    generation_ms: float = 0.0
    repair_generation_ms: float = 0.0
    execution_ms: float = 0.0
    total_ms: float = 0.0


class QueryResponse(BaseModel):
    """The answer, the SQL behind it, and how it was arrived at."""

    question: str
    ok: bool = Field(..., description="True when SQL executed successfully.")
    sql: str | None = Field(None, description="The statement that ran, or the "
                                              "last one attempted.")
    columns: list[str] = []
    rows: list[list[Any]] = []
    row_count: int | None = None
    truncated: bool = Field(
        False, description="True when more rows matched than were returned.")

    repaired: bool = Field(False, description="A repair attempt was used.")
    attempts: list[AttemptInfo] = []
    error: str | None = Field(None, description="Why it failed, in plain text.")
    error_stage: str | None = None

    timings: Timings = Timings()
    model: str | None = None


class SchemaResponse(BaseModel):
    """What the model is told about the database."""

    fingerprint: str
    characters: int
    tables: list[str]
    schema_text: str


class HealthResponse(BaseModel):
    """Liveness and readiness in one place."""

    status: str = Field(..., description="'ok' or 'degraded'.")
    database: bool
    model: bool
    model_id: str | None = None
    schema_fingerprint: str | None = None
    prompt_fingerprint: str | None = None
    detail: str | None = None
