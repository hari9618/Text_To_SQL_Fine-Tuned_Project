"""FastAPI service exposing the Text-to-SQL pipeline.

Phase 12.

    POST /query    question in, rows plus the SQL that produced them out
    GET  /schema   what the model is told about the database
    GET  /health   database and model readiness
    GET  /docs     interactive OpenAPI (FastAPI built-in)

Run it (from the project root, so ``src`` is importable):

    env\\Scripts\\python.exe -m uvicorn src.api.main:app --reload

Design notes worth stating, because they are the difference between a demo and
a service:

*The model and the schema are built once, at start-up.* Loading them per
request would add seconds to every call and re-read a schema that cannot change
without a restart anyway.

*The connection pool is read-only and time-limited.* Every connection is
configured with ``default_transaction_read_only`` and a statement timeout
before it is handed out, so a query that escapes validation still cannot write
and still cannot hang the server.

*A bad question is not a server error.* Generated SQL that fails to parse or
fails to execute returns HTTP 200 with ``ok: false`` and the reason. Reserving
5xx for genuine server faults is what makes the error rate meaningful.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from src.api.backends import build_model
from src.api.schemas import (
    AttemptInfo,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SchemaResponse,
    Timings,
)
from src.api.service import TextToSQLService
from src.model.prompt import PROMPT_VERSION, prompt_fingerprint
from src.sql.config import ConfigError, app_config
from src.sql.executor import DEFAULT_STATEMENT_TIMEOUT_MS

log = logging.getLogger("text2sql.api")

STATEMENT_TIMEOUT_MS = int(
    os.getenv("SQL_STATEMENT_TIMEOUT_MS", str(DEFAULT_STATEMENT_TIMEOUT_MS)))


def _configure_connection(conn) -> None:
    """Make every pooled connection read-only and time-limited.

    Applied on checkout rather than once at creation: a pooled connection is
    reused across requests, and a session setting reset by anything at all
    would silently remove the guarantee.
    """
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
        cur.execute("SET default_transaction_read_only = on")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the pool, the model and the schema context once."""
    from psycopg_pool import ConnectionPool

    app.state.ready = False
    app.state.startup_error = None
    app.state.pool = None
    app.state.service = None

    try:
        cfg = app_config()
    except ConfigError as exc:
        # Start anyway so /health can explain what is wrong. Refusing to boot
        # makes a misconfigured deployment look like a crash loop.
        app.state.startup_error = str(exc)
        log.error("configuration error: %s", exc)
        yield
        return

    try:
        pool = ConnectionPool(
            cfg.conninfo(),
            min_size=int(os.getenv("DB_POOL_MIN", "1")),
            max_size=int(os.getenv("DB_POOL_MAX", "4")),
            configure=_configure_connection,
            open=True,
            timeout=10.0,
        )
        app.state.pool = pool

        model = build_model()
        with pool.connection() as conn:
            app.state.service = TextToSQLService.from_connection(model, conn)
        app.state.ready = True
        log.info("ready: model=%s schema=%s",
                 model.model_id, app.state.service.schema_fingerprint)
    except Exception as exc:  # noqa: BLE001 - reported through /health
        app.state.startup_error = f"{type(exc).__name__}: {exc}"
        log.exception("startup failed")

    yield

    if app.state.pool is not None:
        app.state.pool.close()


app = FastAPI(
    title="Enterprise Text-to-SQL",
    version="1.0.0",
    summary="Natural-language questions answered as executed PostgreSQL.",
    description=(
        "Converts business questions into PostgreSQL, validates the SQL, runs "
        "it against a read-only session and returns the rows. On a detectable "
        "failure it feeds the database error back to the model and retries "
        "once.\n\n"
        "Accuracy depends on the configured backend: the base model scores "
        "10.82 % strict execution accuracy on the project benchmark, the "
        "fine-tuned adapter 50.99 %."
    ),
    lifespan=lifespan,
)


def _service(request: Request) -> TextToSQLService:
    """The live service, or a 503 explaining why there is not one."""
    service = getattr(request.app.state, "service", None)
    if service is None or not request.app.state.ready:
        raise HTTPException(
            status_code=503,
            detail=request.app.state.startup_error or "service is starting",
        )
    return service


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    """Readiness, including whether the database actually answers.

    Deliberately executes ``SELECT 1`` rather than trusting that a pool exists.
    A pool object with no reachable database is exactly the failure a health
    check is meant to catch.
    """
    state = request.app.state
    db_ok = False
    detail = state.startup_error

    pool = getattr(state, "pool", None)
    if pool is not None:
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            db_ok = True
        except Exception as exc:  # noqa: BLE001
            detail = f"database unreachable: {type(exc).__name__}"

    service = getattr(state, "service", None)
    return HealthResponse(
        status="ok" if (db_ok and service is not None) else "degraded",
        database=db_ok,
        model=service is not None,
        model_id=service.model.model_id if service else None,
        schema_fingerprint=service.schema_fingerprint if service else None,
        prompt_fingerprint=prompt_fingerprint(),
        detail=detail,
    )


@app.get("/schema", response_model=SchemaResponse, tags=["introspection"])
def schema(request: Request) -> SchemaResponse:
    """Exactly what the model is shown. Useful when a result looks wrong."""
    service = _service(request)
    return SchemaResponse(
        fingerprint=service.schema_fingerprint,
        characters=len(service.schema_text),
        tables=sorted(service.schema_info.tables),
        schema_text=service.schema_text,
    )


@app.post("/query", response_model=QueryResponse, tags=["query"])
def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Answer one question.

    Returns 200 even when the generated SQL fails; ``ok`` carries that. 5xx is
    reserved for the service itself being broken.
    """
    service = _service(request)
    pool = request.app.state.pool

    started = time.perf_counter()
    try:
        with pool.connection() as conn:
            result = service.answer(
                body.question, conn,
                max_rows=body.max_rows,
                repair=body.repair,
                include_rows=body.include_rows,
            )
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline failure")
        raise HTTPException(
            status_code=503,
            detail=f"pipeline unavailable: {type(exc).__name__}",
        ) from exc

    return QueryResponse(
        question=body.question,
        ok=result.ok,
        sql=result.sql,
        columns=result.columns,
        rows=result.rows,
        row_count=result.row_count,
        truncated=result.truncated,
        repaired=result.repaired,
        attempts=[AttemptInfo(**a) for a in result.attempts],
        error=result.error,
        error_stage=result.error_stage,
        timings=Timings(
            generation_ms=result.generation_ms,
            repair_generation_ms=result.repair_generation_ms,
            execution_ms=result.execution_ms,
            total_ms=round((time.perf_counter() - started) * 1000, 2),
        ),
        model=service.model.model_id,
    )


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Never leak a traceback, a file path or a connection string to a caller."""
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error", "type": type(exc).__name__},
    )


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def root() -> Response:
    """The demo UI, or a JSON descriptor if the static file is absent.

    Falling back rather than 404-ing matters for the container: the API is
    useful headless, and a missing `static/` should not take the service down.
    """
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return JSONResponse({
        "service": "Enterprise Text-to-SQL",
        "docs": "/docs",
        "health": "/health",
        "prompt_version": PROMPT_VERSION,
    })
