"""Benchmark runner: score a model on the held-out test split.

Phase 4.

Per example:

    question + full schema
            |
            v
      model generates SQL          <- network, timed separately
            |
            v
      static validation            <- parseable? read-only? real tables?
            |
            v
      execute on PostgreSQL        <- read-only session, 30 s timeout
            |
            v
      compare result fingerprints  <- never compares SQL text
            |
            v
      one outcome + latencies

Two design choices worth stating.

*Generation runs concurrently, execution runs sequentially.* Generation is
network-bound and dominates wall-clock; execution touches one database
connection, which is not thread-safe. Each call still times itself, so
concurrency does not distort the latency figures.

*Gold queries are re-executed.* The stored Phase 3 fingerprints could have
been produced against different data. Re-running gold in the same session
proves the database still matches, and yields the order-sensitive fingerprint
needed for the strict metric. A mismatch is reported loudly rather than
silently scored.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import psycopg

from src.constants import DATA_AS_OF, DATA_AS_OF_SQL
from src.evaluation.metrics import (
    ExampleResult,
    Outcome,
    outcome_from_error_class,
)
from src.model.base_model import TextToSQLModel
from src.model.prompt import PROMPT_VERSION, prompt_fingerprint
from src.sql import validator
from src.sql.executor import ExecutionResult, SchemaInfo, execute


def load_examples(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return rows[:limit] if limit else rows


def stride_sample(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Pick n examples spread evenly through the split.

    The test split is ordered by template, so taking the first n would return n
    near-identical paraphrases of a single query -- a smoke test that passes or
    fails on one template tells you almost nothing. Striding covers the widest
    span of templates and difficulties for a given n, and is deterministic, so
    the same subset is used every run.
    """
    if n >= len(rows):
        return rows
    step = len(rows) / n
    return [rows[min(int(i * step), len(rows) - 1)] for i in range(n)]


def dataset_fingerprint(path: Path) -> str:
    """Hash of the split file. Detects a changed test set between runs."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def database_fingerprint(conn: psycopg.Connection, tables: Iterable[str]) -> str:
    """Hash of row counts per table.

    Cheap proof that two runs saw the same data. It would not catch an edit
    that preserved row counts, but it reliably catches a regenerated database —
    the realistic risk here.
    """
    counts = []
    with conn.cursor() as cur:
        for table in sorted(tables):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            counts.append(f"{table}={cur.fetchone()[0]}")
    joined = ";".join(counts)
    return hashlib.sha256(joined.encode()).hexdigest()[:16], joined


@dataclass
class GoldExecution:
    """Result of re-running one gold query."""

    fingerprint: str
    ordered_fingerprint: str
    row_count: int
    matches_stored: bool
    columns: tuple[str, ...] = ()


def execute_gold(
    conn: psycopg.Connection, examples: list[dict[str, Any]]
) -> tuple[dict[str, GoldExecution], list[str]]:
    """Re-execute each distinct gold query; report any fingerprint drift."""
    cache: dict[str, GoldExecution] = {}
    drift: list[str] = []

    stored_by_sql: dict[str, str] = {}
    for ex in examples:
        stored_by_sql.setdefault(ex["sql"], ex["execution"]["fingerprint"])

    for sql, stored in stored_by_sql.items():
        result = execute(conn, sql)
        if not result.ok:
            drift.append(f"gold query failed to execute: {result.error_message}")
            continue
        matches = result.fingerprint == stored
        if not matches:
            drift.append(
                f"stored fingerprint {stored} != recomputed {result.fingerprint} "
                f"for: {sql[:90]}"
            )
        cache[sql] = GoldExecution(
            fingerprint=result.fingerprint or "",
            ordered_fingerprint=result.ordered_fingerprint or "",
            row_count=result.row_count or 0,
            matches_stored=matches,
            columns=result.columns,
        )
    return cache, drift


def _classify_and_execute(
    conn: psycopg.Connection,
    schema_info: SchemaInfo,
    predicted_sql: str,
) -> tuple[Outcome, ExecutionResult | None, str | None]:
    """Static checks then execution. Returns (outcome, result, error message).

    Static checks run first so that a hallucinated table is reported as
    hallucination rather than as whatever PostgreSQL happens to complain about,
    and so a destructive statement is never sent to the database at all.
    """
    static = validator.analyse(predicted_sql, schema_info.tables)

    if not static.parseable:
        return Outcome.INVALID_SQL, None, static.parse_error
    if static.statement_count != 1:
        return (Outcome.INVALID_SQL, None,
                f"{static.statement_count} statements in one answer")
    if not static.is_read_only:
        return (Outcome.NOT_READ_ONLY, None,
                f"statement kind: {static.statement_kind}")
    if static.unknown_tables:
        return (Outcome.UNKNOWN_TABLE, None,
                f"unknown tables: {', '.join(static.unknown_tables)}")

    result = execute(conn, predicted_sql)
    if not result.ok:
        return (outcome_from_error_class(result.error_class), result,
                result.error_message)
    return Outcome.CORRECT, result, None  # provisional; caller compares results


def projection_matches(
    conn: psycopg.Connection,
    predicted_sql: str,
    gold: GoldExecution,
) -> bool:
    """Are the gold rows recoverable from the prediction by column selection?

    Many benchmark questions never state which columns to return -- "Show
    orders placed in 2023" is answered just as well by SELECT * as by the four
    columns the gold query happens to name. Strict fingerprint comparison marks
    those wrong, which measures the benchmark's arbitrary column choice rather
    than the model's SQL.

    Rather than pulling both result sets into memory and diffing them, the
    prediction is wrapped in a subquery and PostgreSQL is asked for the gold
    columns. If they exist and the resulting rows hash to the gold
    fingerprint, the prediction contained the right answer. Memory stays flat
    even for a 12,000-row result.

    This is a *diagnostic*. It never changes the headline execution accuracy.
    """
    if not gold.columns:
        return False
    projection = ", ".join('"' + c.replace('"', '""') + '"' for c in gold.columns)
    probe = f"SELECT {projection} FROM ({predicted_sql}) AS _pred_sub"
    result = execute(conn, probe)
    return bool(result.ok and result.fingerprint == gold.fingerprint)


def score_example(
    conn: psycopg.Connection,
    schema_info: SchemaInfo,
    example: dict[str, Any],
    generation,
    gold: GoldExecution | None,
) -> ExampleResult:
    """Turn one generation into a scored result."""
    base = dict(
        example_id=example["id"],
        template_id=example["template_id"],
        difficulty=example["difficulty"],
        domain=example["domain"],
        query_type=list(example.get("query_type", [])),
        question=example["question"],
        gold_sql=example["sql"],
        predicted_sql=generation.sql,
        generation_latency_ms=generation.latency_ms,
        gold_fingerprint=gold.fingerprint if gold else None,
        gold_row_count=gold.row_count if gold else None,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        attempts=getattr(generation, "attempts", 1),
        raw_output=generation.raw_output[:4000],
    )

    if not generation.ok:
        return ExampleResult(
            **base, outcome=Outcome.GENERATION_FAILED, correct=False,
            total_latency_ms=generation.latency_ms,
            error_message=generation.error,
        )

    if not generation.sql.strip():
        return ExampleResult(
            **base, outcome=Outcome.NO_SQL_PRODUCED, correct=False,
            total_latency_ms=generation.latency_ms,
            error_message="model reply contained no SELECT statement",
        )

    outcome, result, error = _classify_and_execute(
        conn, schema_info, generation.sql
    )
    exec_ms = result.latency_ms if result else 0.0

    if outcome is not Outcome.CORRECT:
        return ExampleResult(
            **base, outcome=outcome, correct=False,
            execution_latency_ms=exec_ms,
            total_latency_ms=generation.latency_ms + exec_ms,
            error_message=error,
            predicted_row_count=result.row_count if result else None,
        )

    assert result is not None
    correct = bool(gold and result.fingerprint == gold.fingerprint)

    # Only probed when the strict comparison failed, so the extra query costs
    # nothing on correct answers.
    projection_ok: bool | None = None
    if gold and not correct:
        projection_ok = projection_matches(conn, generation.sql, gold)

    # Order matters only where the gold query asked for it.
    strict: bool | None = None
    if gold and "ORDER BY" in example["sql"].upper():
        strict = result.ordered_fingerprint == gold.ordered_fingerprint

    return ExampleResult(
        **base,
        outcome=Outcome.CORRECT if correct else Outcome.WRONG_RESULT,
        correct=correct,
        strict_correct=strict,
        projection_match=projection_ok,
        execution_latency_ms=exec_ms,
        total_latency_ms=generation.latency_ms + exec_ms,
        predicted_fingerprint=result.fingerprint,
        predicted_row_count=result.row_count,
    )


def load_generation_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Read previously completed generations, keyed by example id."""
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                cache[row["example_id"]] = row
    return cache


def run_benchmark(
    model: TextToSQLModel,
    examples: list[dict[str, Any]],
    conn: psycopg.Connection,
    schema_text: str,
    schema_info: SchemaInfo,
    workers: int = 4,
    progress: Callable[[int, int], None] | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    fatal_abort_threshold: int = 25,
    schema_for: Callable[[dict[str, Any]], tuple[str, dict[str, Any] | None]] | None = None,
) -> tuple[list[ExampleResult], list[str]]:
    """Generate for every example, then score each against the gold result.

    Generations are checkpointed to ``checkpoint_path`` as they complete. A
    453-example run takes many minutes over the network; without this, a
    provider outage or an exhausted quota partway through throws away every
    successful call made so far. ``resume`` replays the checkpoint and only
    generates what is missing.

    ``fatal_abort_threshold`` stops the run after that many unrecoverable
    errors (exhausted credits, bad token). Continuing past those produces
    hundreds of identical failures that look like model errors in the report.
    """
    import threading

    gold_cache, drift = execute_gold(conn, examples)

    cached = load_generation_cache(checkpoint_path) if (resume and checkpoint_path) else {}
    if cached:
        print(f"  resuming: {len(cached):,} generations already on disk")

    abort = threading.Event()
    fatal_count = 0
    write_lock = threading.Lock()
    fh = checkpoint_path.open("a", encoding="utf-8") if checkpoint_path else None

    def record(example_id: str, gen) -> None:
        if fh is None:
            return
        with write_lock:
            fh.write(json.dumps({"example_id": example_id, **gen.as_dict()},
                                ensure_ascii=False) + chr(10))
            fh.flush()

    def generate(example: dict[str, Any]):
        nonlocal fatal_count
        hit = cached.get(example["id"])
        if hit is not None:
            from src.model.base_model import GenerationResult
            return GenerationResult(
                sql=hit.get("sql", ""), raw_output=hit.get("raw_output", ""),
                latency_ms=hit.get("latency_ms", 0.0), ok=hit.get("ok", False),
                error=hit.get("error"), finish_reason=hit.get("finish_reason"),
                prompt_tokens=hit.get("prompt_tokens"),
                completion_tokens=hit.get("completion_tokens"),
                provider=hit.get("provider"), attempts=hit.get("attempts", 1),
                fatal=hit.get("fatal", False),
            )
        if abort.is_set():
            from src.model.base_model import GenerationResult
            return GenerationResult(
                sql="", raw_output="", latency_ms=0.0, ok=False,
                error="run aborted after repeated unrecoverable errors",
                fatal=True,
            )
        prompt_schema = (schema_for(example)[0] if schema_for else schema_text)
        gen = model.generate(example["question"], prompt_schema)
        if getattr(gen, "fatal", False):
            with write_lock:
                fatal_count += 1
                if fatal_count >= fatal_abort_threshold:
                    abort.set()
        record(example["id"], gen)
        return gen

    generations: list[Any] = [None] * len(examples)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(generate, ex): i for i, ex in enumerate(examples)
            }
            for done, future in enumerate(as_completed(futures), start=1):
                generations[futures[future]] = future.result()
                if progress:
                    progress(done, len(examples))
    else:
        for i, example in enumerate(examples):
            generations[i] = generate(example)
            if progress:
                progress(i + 1, len(examples))

    if fh is not None:
        fh.close()

    if abort.is_set():
        drift.append(
            f"RUN ABORTED: {fatal_count} unrecoverable generation errors "
            "(exhausted credits, bad token, or no model access). Results are "
            "partial and must not be reported as a baseline."
        )

    results = []
    for example, generation in zip(examples, generations):
        scored = score_example(conn, schema_info, example, generation,
                               gold_cache.get(example["sql"]))
        if schema_for:
            _, meta = schema_for(example)
            if meta is not None:
                # Did the model cite a real table it was never shown? That is
                # not hallucination -- the table exists -- but it does mean the
                # answer could only have come from prior knowledge of the
                # schema, so retrieval failed to supply what was needed.
                shown = set(meta.get("tables", []))
                static = validator.analyse(scored.predicted_sql, schema_info.tables)
                unseen = sorted(set(static.referenced_tables) - shown)
                meta = {**meta, "used_unretrieved_tables": unseen}
            scored.retrieval = meta
        results.append(scored)
    return results, drift


def build_run_metadata(
    model: TextToSQLModel,
    dataset_path: Path,
    conn: psycopg.Connection,
    schema_fingerprint_value: str,
    schema_text: str,
    tables: Iterable[str],
) -> dict[str, Any]:
    """Everything needed to reproduce or invalidate this run."""
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        pg_version = cur.fetchone()[0].split(",")[0]

    db_fp, db_counts = database_fingerprint(conn, tables)

    return {
        "model": model.describe(),
        "prompt": {
            "version": PROMPT_VERSION,
            "fingerprint": prompt_fingerprint(),
        },
        "schema_context": {
            "mode": "full_schema_no_retrieval",
            "fingerprint": schema_fingerprint_value,
            "characters": len(schema_text),
            "note": (
                "Phase 4 is ablation configuration 1: the base model sees the "
                "entire schema. Phase 6 replaces this with a retrieved subset."
            ),
        },
        "dataset": {
            "path": str(dataset_path).replace("\\", "/"),
            "fingerprint": dataset_fingerprint(dataset_path),
        },
        "database": {
            "postgres_version": pg_version,
            "data_fingerprint": db_fp,
            "row_counts": db_counts,
            "data_as_of": DATA_AS_OF.isoformat(),
            "data_as_of_sql": DATA_AS_OF_SQL,
        },
        "evaluation": {
            "method": "execution_based",
            "comparison": "order-insensitive result fingerprint",
            "strict_comparison": "order-sensitive, gold queries with ORDER BY only",
            "sql_string_comparison": False,
        },
    }
