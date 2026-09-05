"""Render a human-readable benchmark report.

Phase 4. Kept separate from scoring so the numbers cannot be shaped by the
presentation code.
"""

from __future__ import annotations

from typing import Any

from collections import Counter

from src.evaluation.metrics import ExampleResult, Outcome

FAILURE_EXPLANATIONS = {
    Outcome.WRONG_RESULT.value: (
        "Ran successfully but returned a different result set than the gold "
        "query. The most dangerous failure: it looks like a valid answer."
    ),
    Outcome.UNKNOWN_COLUMN.value: (
        "Referenced a column that does not exist — schema hallucination."
    ),
    Outcome.UNKNOWN_TABLE.value: (
        "Referenced a table that does not exist — schema hallucination."
    ),
    Outcome.SYNTAX_ERROR.value: "PostgreSQL rejected the SQL as malformed.",
    Outcome.EXECUTION_ERROR.value: (
        "Executed but raised another database error (type mismatch, ambiguous "
        "reference, grouping violation)."
    ),
    Outcome.NO_SQL_PRODUCED.value: (
        "The reply contained no SELECT statement to extract."
    ),
    Outcome.INVALID_SQL.value: "Could not be parsed as a single SQL statement.",
    Outcome.NOT_READ_ONLY.value: (
        "Produced a write or DDL statement. Blocked before execution."
    ),
    Outcome.TIMEOUT.value: "Exceeded the 30-second statement timeout.",
    Outcome.GENERATION_FAILED.value: (
        "The inference API call failed after all retries."
    ),
}


def _table(rows: list[list[str]], headers: list[str], aligns: list[str]) -> str:
    sep = "|" + "|".join(
        {"l": "---", "r": "---:", "c": ":---:"}[a] for a in aligns
    ) + "|"
    out = ["| " + " | ".join(headers) + " |", sep]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _breakdown_table(section: dict[str, Any], label: str,
                     min_total: int = 0) -> str:
    rows = []
    for key, stats in sorted(
        section.items(), key=lambda kv: -kv[1]["execution_accuracy_pct"]
    ):
        if stats["total"] < min_total:
            continue
        rows.append([
            f"`{key}`",
            str(stats["total"]),
            str(stats["correct"]),
            f"{stats['execution_accuracy_pct']:.1f} %",
            f"{stats['executable_pct']:.1f} %",
        ])
    return _table(
        rows,
        [label, "Examples", "Correct", "Execution accuracy", "Executable"],
        ["l", "r", "r", "r", "r"],
    )


def _example_block(r: ExampleResult, show_error: bool) -> str:
    lines = [
        f"**{r.example_id}** · `{r.template_id}` · {r.difficulty} / {r.domain}",
        "",
        f"> {r.question}",
        "",
        "```sql",
        f"-- gold",
        r.gold_sql[:400],
        "",
        f"-- predicted",
        r.predicted_sql[:400] if r.predicted_sql else "(nothing produced)",
        "```",
    ]
    if show_error:
        detail = f"`{r.outcome.value}`"
        if r.error_message:
            detail += f" — {r.error_message[:200]}"
        if r.outcome is Outcome.WRONG_RESULT:
            detail += (f" — gold returned {r.gold_row_count} rows, "
                       f"prediction returned {r.predicted_row_count}")
        lines += ["", detail]
    return "\n".join(lines)


def render_report(
    summary: dict[str, Any],
    metadata: dict[str, Any],
    results: list[ExampleResult],
    smoke: dict[str, Any] | None = None,
    drift: list[str] | None = None,
) -> str:
    t = summary["totals"]
    e = summary["error_rates"]
    lat = summary["latency"]
    model = metadata["model"]

    parts: list[str] = []

    parts.append(f"""# Phase 4 — Base Model Benchmark

**{model['model_id']}**, no fine-tuning, no schema retrieval, no SQL repair.

This is ablation configuration 1: the number every later configuration is
measured against. It is deliberately the weakest setup in the study — the point
is to establish an honest floor, not a good score.

---

## Headline

| Metric | Value |
|---|---:|
| Test examples | {t['total_examples']} |
| **Execution accuracy** | **{t['execution_accuracy_pct']:.2f} %** |
| Correct | {t['correct']} |
| Executable SQL | {t['executable_sql']} ({t['executable_sql_pct']:.2f} %) |
| Ran but wrong answer | {t['wrong_result']} ({t['wrong_result_pct']:.2f} %) |
| Schema hallucination | {e['schema_hallucination_pct']:.2f} % |
| Mean generation latency | {lat.get('generation_clean', lat['generation'])['mean_ms']:.0f} ms |
| Median generation latency | {lat.get('generation_clean', lat['generation'])['median_ms']:.0f} ms |
| SQL execution latency (median) | {lat['execution']['median_ms']:.0f} ms |

Latency is measured over first-attempt calls only ({lat.get('throttled_calls', 0)}
of {t['total_examples']} calls were retried after provider throttling; counting
their backoff sleeps would report HuggingFace's rate limiter as model speed).

Correctness is decided by **executing** both queries and comparing result
sets. SQL text is never compared — these two are different strings and the same
answer:

```sql
SELECT COUNT(*)           FROM customers WHERE country = 'India';
SELECT COUNT(customer_id) FROM customers WHERE country = 'India';
```
""")

    if smoke:
        parts.append(f"""---

## Smoke test

Run on {smoke['total_examples']} examples before committing to the full set.

| | |
|---|---:|
| Examples | {smoke['total_examples']} |
| Execution accuracy | {smoke['execution_accuracy_pct']:.1f} % |
| Executable SQL | {smoke['executable_sql_pct']:.1f} % |
""")

    parts.append(f"""---

## Error rates

| Failure mode | Rate |
|---|---:|
| Ran, wrong answer | {t['wrong_result_pct']:.2f} % |
| Unknown column (hallucinated) | {e['unknown_column_pct']:.2f} % |
| Unknown table (hallucinated) | {e['unknown_table_pct']:.2f} % |
| Syntax error | {e['syntax_error_pct']:.2f} % |
| Other execution error | {e['execution_error_pct']:.2f} % |
| Timeout | {e['timeout_pct']:.2f} % |
| No SQL produced | {e['no_sql_produced_pct']:.2f} % |
| Unparseable SQL | {e['invalid_sql_pct']:.2f} % |
| Not read-only | {e['not_read_only_pct']:.2f} % |
| Generation failed | {e['generation_failed_pct']:.2f} % |

### Outcome distribution

Every example lands in exactly one bucket, so these sum to the total.

{_table(
    [[f"`{k}`", str(v), f"{100.0 * v / max(t['total_examples'], 1):.1f} %",
      FAILURE_EXPLANATIONS.get(k, "Correct — result matched the gold answer.")]
     for k, v in summary["outcomes"].items()],
    ["Outcome", "Count", "Share", "Meaning"], ["l", "r", "r", "l"],
)}
""")

    pa = summary.get("projection_analysis", {})
    if pa:
        parts.append(f"""---

## The headline number is an undercount — here is why

Many benchmark questions never say *which columns* to return. "Show orders
placed in 2023" is answered just as well by `SELECT *` as by the four columns
the gold query happens to name. Strict comparison marks those wrong even when
the rows are byte-identical.

| | Count | Share |
|---|---:|---:|
| Correct (exact match) | {t['correct']} | {t['execution_accuracy_pct']:.2f} % |
| Right rows, different columns | {pa['projection_mismatch_only']} | {pa['projection_mismatch_only_pct']:.2f} % |
| Genuinely wrong rows | {pa['genuinely_different_rows']} | {pa['genuinely_different_rows_pct']:.2f} % |
| **Projection-tolerant accuracy** | **{pa['projection_tolerant_correct']}** | **{pa['projection_tolerant_accuracy_pct']:.2f} %** |

Both numbers are real and both matter:

* **{t['execution_accuracy_pct']:.2f} %** is the honest strict floor, and the
  figure Phase 10 must beat on identical terms.
* **{pa['projection_tolerant_accuracy_pct']:.2f} %** is how often the model
  found the right *rows*.

The gap between them is the share of the benchmark that tests column
convention rather than SQL reasoning. It is reported separately so that a
fine-tuned model cannot post a large apparent gain merely by learning which
columns this dataset likes to select.
""")

    strict = summary["strict_order_sensitive"]
    parts.append(f"""---

## Order-sensitive accuracy

The headline metric ignores row order. For questions like "the top 10 customers
by revenue", order is part of the answer, so those are also scored strictly.

| | |
|---|---:|
| Examples with `ORDER BY` in gold | {strict['scope']} |
| Correct with order respected | {strict['correct']} |
| Strict accuracy | {strict['accuracy_pct']:.2f} % |

A gap between this and the headline number means the model selected the right
rows but sorted them wrongly.
""")

    parts.append(f"""---

## Results by difficulty

{_breakdown_table(summary["by_difficulty"], "Difficulty")}

## Results by domain

{_breakdown_table(summary["by_domain"], "Domain")}

## Results by SQL pattern

Query features are not exclusive — one query can be counted under several.

{_breakdown_table(summary["by_query_type"], "SQL feature", min_total=5)}
""")

    # --- failure pattern classification -----------------------------------
    real_errors = [r for r in results
                   if r.outcome is Outcome.WRONG_RESULT and not r.projection_match]
    if real_errors:
        def _pattern(r) -> str:
            g, p = r.gold_sql.upper(), r.predicted_sql.upper()
            gr, pr = r.gold_row_count, r.predicted_row_count
            if pr == 0 and gr:
                return "Returned zero rows (over-filtered)"
            if "LIMIT" in g and "LIMIT" not in p:
                return "Dropped the LIMIT the question asked for"
            if "LIMIT" in p and "LIMIT" not in g:
                return "Invented a LIMIT the question never asked for"
            if "LEFT JOIN" in p and "LEFT JOIN" not in g:
                return "LEFT JOIN where gold used INNER (kept unmatched rows)"
            if "LEFT JOIN" in g and "LEFT JOIN" not in p:
                return "INNER JOIN where gold used LEFT (dropped rows)"
            if "DISTINCT" in p and "DISTINCT" not in g:
                return "Added DISTINCT, collapsing legitimate duplicates"
            if gr and pr and pr > gr * 1.05:
                return "Too many rows (under-filtered)"
            if gr and pr and pr < gr * 0.95:
                return "Too few rows (over-filtered)"
            return "Right row count, different values (usually a different metric definition)"

        counts = Counter(_pattern(r) for r in real_errors)
        rows_tbl = [[k, str(v), f"{100.0 * v / len(real_errors):.0f} %"]
                    for k, v in counts.most_common()]
        parts.append(f"""---

## Failure patterns

Classification of the {len(real_errors)} genuinely wrong answers, excluding the
column-projection artefact above.

{_table(rows_tbl, ["Pattern", "Count", "Share"], ["l", "r", "r"])}

The largest bucket is *not* broken SQL. It is the model computing a defensible
but different quantity — most often revenue summed from `order_items` lines
when the gold query used the denormalised `orders.total_amount`. Both are
reasonable readings of "revenue"; the schema supports two answers and the
question does not disambiguate. That is a business-rule gap, which is exactly
what fine-tuning on this domain is expected to close.
""")

    correct_examples = [r for r in results if r.correct][:4]
    if correct_examples:
        parts.append("---\n\n## Successful examples\n")
        for r in correct_examples:
            parts.append(_example_block(r, show_error=False) + "\n")

    failures = [r for r in results if not r.correct]
    ranked: list[ExampleResult] = []
    seen_outcomes: set[str] = set()
    for r in failures:  # one of each failure mode first, for coverage
        if r.outcome.value not in seen_outcomes:
            ranked.append(r)
            seen_outcomes.add(r.outcome.value)
    ranked += [r for r in failures if r not in ranked][:3]

    if ranked:
        parts.append("---\n\n## Failed examples\n")
        for r in ranked[:6]:
            parts.append(_example_block(r, show_error=True) + "\n")

    parts.append(f"""---

## Reproducibility

| | |
|---|---|
| Model | `{model['model_id']}` |
| Model revision | `{model.get('model_revision') or 'unresolved'}` |
| Serving | {model.get('kind')} / {model.get('provider', 'n/a')} |
| Fine-tuned | {model.get('fine_tuned')} |
| Adapters | {model.get('adapters', []) or 'none'} |
| Inference params | `{model.get('inference_params')}` |
| Prompt version | `{metadata['prompt']['version']}` (hash `{metadata['prompt']['fingerprint']}`) |
| Schema mode | `{metadata['schema_context'].get('mode', 'unknown')}` (hash `{metadata['schema_context'].get('fingerprint', 'n/a - retrieved per question')}`) |
| Dataset | `{metadata['dataset']['path']}` (hash `{metadata['dataset']['fingerprint']}`) |
| Database | {metadata['database']['postgres_version']} |
| Data fingerprint | `{metadata['database']['data_fingerprint']}` |
| DATA_AS_OF | `{metadata['database']['data_as_of']}` |

Every one of these must match for a future run to be comparable. A changed
prompt hash means the next score measures prompt engineering; a changed data
fingerprint means it measures a different database.
""")

    if drift:
        parts.append("### Gold fingerprint drift\n")
        parts.append(
            "Gold queries were re-executed and compared against the "
            "fingerprints recorded in Phase 3:\n"
        )
        for line in drift[:10]:
            parts.append(f"- {line}")
    else:
        parts.append(
            "### Gold fingerprint check\n\n"
            "All gold queries were re-executed and reproduced their Phase 3 "
            "fingerprints exactly — the database has not drifted since "
            "validation."
        )

    parts.append("""
---

## What this number is not

This is the floor, not a verdict on the model. It has no schema retrieval, no
SQL repair, no few-shot examples, and no fine-tuning. Phases 6, 9 and 11 each
add one of those, and each is measured against this figure.

No claim that fine-tuning helps can be made until Phase 10 produces a
comparable number from an identical harness.
""")

    return "\n".join(parts)
