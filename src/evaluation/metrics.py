"""Outcome classification and metric aggregation.

Phase 4.

Correctness is decided by **execution**, never by comparing SQL strings. These
two queries are different text and the same answer, and both must count as
correct:

    SELECT COUNT(*)           FROM customers WHERE country = 'India';
    SELECT COUNT(customer_id) FROM customers WHERE country = 'India';

Every example therefore lands in exactly one outcome, so the categories sum to
the total and no failure can be double-counted or silently dropped.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class Outcome(str, Enum):
    """Mutually exclusive result of one example."""

    CORRECT = "correct"
    # Ran, returned something, but not the gold answer. The most dangerous
    # failure in production: it looks like a successful answer.
    WRONG_RESULT = "wrong_result"

    GENERATION_FAILED = "generation_failed"   # the API call itself failed
    NO_SQL_PRODUCED = "no_sql_produced"       # reply contained no statement
    INVALID_SQL = "invalid_sql"               # unparseable
    NOT_READ_ONLY = "not_read_only"           # INSERT/UPDATE/DELETE/DDL
    UNKNOWN_TABLE = "unknown_table"           # hallucinated table
    UNKNOWN_COLUMN = "unknown_column"         # hallucinated column
    SYNTAX_ERROR = "syntax_error"             # PostgreSQL rejected it
    EXECUTION_ERROR = "execution_error"       # ran but errored otherwise
    TIMEOUT = "timeout"


# Outcomes that count as schema hallucination — the model invented a table or
# column that does not exist. Tracked separately because it is the specific
# failure schema retrieval (Phase 6) and fine-tuning (Phase 9) should reduce.
HALLUCINATION_OUTCOMES = {Outcome.UNKNOWN_TABLE, Outcome.UNKNOWN_COLUMN}

# Outcomes where the SQL never reached a successful execution.
NON_EXECUTABLE_OUTCOMES = {
    Outcome.GENERATION_FAILED, Outcome.NO_SQL_PRODUCED, Outcome.INVALID_SQL,
    Outcome.NOT_READ_ONLY, Outcome.UNKNOWN_TABLE, Outcome.UNKNOWN_COLUMN,
    Outcome.SYNTAX_ERROR, Outcome.EXECUTION_ERROR, Outcome.TIMEOUT,
}


def outcome_from_error_class(error_class: str | None) -> Outcome:
    """Map a PostgreSQL error class from the executor onto an outcome."""
    return {
        "undefined_table": Outcome.UNKNOWN_TABLE,
        "undefined_column": Outcome.UNKNOWN_COLUMN,
        "syntax_error": Outcome.SYNTAX_ERROR,
        "timeout": Outcome.TIMEOUT,
    }.get(error_class or "", Outcome.EXECUTION_ERROR)


@dataclass
class ExampleResult:
    """Everything recorded for one benchmark example."""

    example_id: str
    template_id: str
    difficulty: str
    domain: str
    query_type: list[str]
    question: str
    gold_sql: str
    predicted_sql: str
    outcome: Outcome
    correct: bool
    strict_correct: bool | None = None
    # Set only when a query ran but did not match strictly. True means it
    # returned the gold rows but with a different column projection -- most
    # often SELECT * where the gold named specific columns. That is a benchmark
    # artefact, not a SQL error, so it is kept apart from genuine wrong answers.
    projection_match: bool | None = None
    generation_latency_ms: float = 0.0
    execution_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    gold_fingerprint: str | None = None
    predicted_fingerprint: str | None = None
    gold_row_count: int | None = None
    predicted_row_count: int | None = None
    error_message: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # >1 means the provider throttled and the call was retried. Latency for
    # these is not a clean measurement of model speed.
    attempts: int = 1
    # Populated only when schema retrieval is active: which tables were shown,
    # whether they covered the gold query, and whether the model referenced a
    # real table it was never shown.
    retrieval: dict[str, Any] | None = None
    raw_output: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items()}
        data["outcome"] = self.outcome.value
        for key in ("generation_latency_ms", "execution_latency_ms",
                    "total_latency_ms"):
            data[key] = round(data[key], 2)
        return data


def _percent(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def _latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p90_ms": 0.0,
                "min_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(values)
    return {
        "mean_ms": round(statistics.fmean(values), 2),
        "median_ms": round(statistics.median(values), 2),
        "p90_ms": round(ordered[min(int(0.9 * len(ordered)), len(ordered) - 1)], 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
    }


@dataclass
class Breakdown:
    total: int = 0
    correct: int = 0
    executable: int = 0
    outcomes: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "execution_accuracy_pct": _percent(self.correct, self.total),
            "executable": self.executable,
            "executable_pct": _percent(self.executable, self.total),
            "outcomes": dict(self.outcomes.most_common()),
        }


def _group(results: Iterable[ExampleResult], key) -> dict[str, dict[str, Any]]:
    grouped: dict[str, Breakdown] = defaultdict(Breakdown)
    for r in results:
        for value in key(r):
            b = grouped[value]
            b.total += 1
            b.correct += int(r.correct)
            b.executable += int(r.outcome not in NON_EXECUTABLE_OUTCOMES)
            b.outcomes[r.outcome.value] += 1
    return {k: v.as_dict() for k, v in sorted(grouped.items())}


def _retrieval_summary(results: list[ExampleResult]) -> dict[str, Any] | None:
    """Retrieval quality, and how accuracy splits on it.

    The decisive comparison is accuracy *when retrieval succeeded* versus *when
    it dropped a needed table*. If the second is far worse, retrieval recall is
    the bottleneck and improving the retriever is worth more than any prompt or
    model change.
    """
    scoped = [r for r in results if r.retrieval]
    if not scoped:
        return None

    complete = [r for r in scoped if r.retrieval.get("complete")]
    incomplete = [r for r in scoped if not r.retrieval.get("complete")]
    used_unseen = sum(
        1 for r in scoped if r.retrieval.get("used_unretrieved_tables")
    )
    tables = [r.retrieval.get("n_tables", 0) for r in scoped]
    chars = [r.retrieval.get("schema_chars", 0) for r in scoped]

    return {
        "examples": len(scoped),
        "complete_retrieval": len(complete),
        "complete_retrieval_pct": _percent(len(complete), len(scoped)),
        "table_recall_pct": _percent(
            sum(r.retrieval.get("gold_found", 0) for r in scoped),
            sum(r.retrieval.get("gold_total", 0) for r in scoped),
        ),
        "mean_tables_shown": round(statistics.fmean(tables), 2) if tables else 0,
        "mean_schema_chars": round(statistics.fmean(chars), 1) if chars else 0,
        "referenced_a_table_never_shown": used_unseen,
        "referenced_a_table_never_shown_pct": _percent(used_unseen, len(scoped)),
        "accuracy_when_retrieval_complete_pct": _percent(
            sum(1 for r in complete if r.correct), len(complete)),
        "accuracy_when_retrieval_incomplete_pct": _percent(
            sum(1 for r in incomplete if r.correct), len(incomplete)),
        "note": (
            "Accuracy split by whether retrieval supplied every table the gold "
            "query needs. A large gap means recall, not generation, is the "
            "limiting factor."
        ),
    }


def summarise(results: list[ExampleResult]) -> dict[str, Any]:
    """Aggregate per-example results into the reported metrics."""
    total = len(results)
    outcomes = Counter(r.outcome.value for r in results)

    correct = sum(1 for r in results if r.correct)
    executable = sum(
        1 for r in results if r.outcome not in NON_EXECUTABLE_OUTCOMES
    )
    hallucinated = sum(1 for r in results if r.outcome in HALLUCINATION_OUTCOMES)

    # Strict accuracy is only meaningful where it was evaluated: gold queries
    # with an ORDER BY, where row order is part of the answer.
    strict_scope = [r for r in results if r.strict_correct is not None]
    strict_correct = sum(1 for r in strict_scope if r.strict_correct)

    # "Projection tolerant" = matched strictly, or the gold rows are
    # recoverable from the prediction by selecting the gold columns. Separates
    # "wrote the wrong query" from "chose different columns for a question that
    # never specified any".
    projection_only = sum(1 for r in results if r.projection_match is True)
    different_rows = sum(
        1 for r in results
        if r.outcome is Outcome.WRONG_RESULT and not r.projection_match
    )

    generation = [r.generation_latency_ms for r in results
                  if r.outcome is not Outcome.GENERATION_FAILED]
    execution = [r.execution_latency_ms for r in results
                 if r.execution_latency_ms > 0]
    totals = [r.total_latency_ms for r in results]

    prompt_tokens = [r.prompt_tokens for r in results if r.prompt_tokens]
    completion_tokens = [r.completion_tokens for r in results
                         if r.completion_tokens]

    return {
        "totals": {
            "total_examples": total,
            "correct": correct,
            "execution_accuracy_pct": _percent(correct, total),
            "executable_sql": executable,
            "executable_sql_pct": _percent(executable, total),
            "wrong_result": outcomes.get(Outcome.WRONG_RESULT.value, 0),
            "wrong_result_pct": _percent(
                outcomes.get(Outcome.WRONG_RESULT.value, 0), total),
        },
        "projection_analysis": {
            "note": (
                "Many benchmark questions never say which columns to return "
                "(\"Show orders placed in 2023\"). The headline metric demands "
                "an exact column match, so SELECT * scores wrong even when the "
                "rows are identical. This splits those cases apart."
            ),
            "projection_mismatch_only": projection_only,
            "projection_mismatch_only_pct": _percent(projection_only, total),
            "genuinely_different_rows": different_rows,
            "genuinely_different_rows_pct": _percent(different_rows, total),
            "projection_tolerant_correct": correct + projection_only,
            "projection_tolerant_accuracy_pct": _percent(
                correct + projection_only, total),
        },
        "error_rates": {
            "syntax_error_pct": _percent(
                outcomes.get(Outcome.SYNTAX_ERROR.value, 0), total),
            "unknown_table_pct": _percent(
                outcomes.get(Outcome.UNKNOWN_TABLE.value, 0), total),
            "unknown_column_pct": _percent(
                outcomes.get(Outcome.UNKNOWN_COLUMN.value, 0), total),
            "schema_hallucination_pct": _percent(hallucinated, total),
            "execution_error_pct": _percent(
                outcomes.get(Outcome.EXECUTION_ERROR.value, 0), total),
            "timeout_pct": _percent(outcomes.get(Outcome.TIMEOUT.value, 0), total),
            "no_sql_produced_pct": _percent(
                outcomes.get(Outcome.NO_SQL_PRODUCED.value, 0), total),
            "invalid_sql_pct": _percent(
                outcomes.get(Outcome.INVALID_SQL.value, 0), total),
            "not_read_only_pct": _percent(
                outcomes.get(Outcome.NOT_READ_ONLY.value, 0), total),
            "generation_failed_pct": _percent(
                outcomes.get(Outcome.GENERATION_FAILED.value, 0), total),
        },
        "strict_order_sensitive": {
            "scope": len(strict_scope),
            "correct": strict_correct,
            "accuracy_pct": _percent(strict_correct, len(strict_scope)),
            "note": (
                "Subset whose gold query has ORDER BY, scored with row order "
                "respected. The headline metric ignores row order."
            ),
        },
        "latency": {
            # First-attempt calls only: no retry backoff mixed in, so this is
            # the honest measure of how fast the model answers.
            "generation_clean": _latency_stats(
                [r.generation_latency_ms for r in results
                 if r.attempts == 1 and r.outcome is not Outcome.GENERATION_FAILED]
            ),
            "throttled_calls": sum(1 for r in results if r.attempts > 1),
            "generation": _latency_stats(generation),
            "execution": _latency_stats(execution),
            "total": _latency_stats(totals),
        },
        "tokens": {
            "prompt_tokens_mean": round(statistics.fmean(prompt_tokens), 1)
                                  if prompt_tokens else None,
            "completion_tokens_mean": round(statistics.fmean(completion_tokens), 1)
                                      if completion_tokens else None,
            "prompt_tokens_total": sum(prompt_tokens) if prompt_tokens else None,
            "completion_tokens_total": sum(completion_tokens)
                                       if completion_tokens else None,
        },
        "retrieval": _retrieval_summary(results),
        "outcomes": dict(outcomes.most_common()),
        "by_difficulty": _group(results, lambda r: [r.difficulty]),
        "by_domain": _group(results, lambda r: [r.domain]),
        "by_query_type": _group(results, lambda r: r.query_type or ["unspecified"]),
        "by_template": _group(results, lambda r: [r.template_id]),
    }
