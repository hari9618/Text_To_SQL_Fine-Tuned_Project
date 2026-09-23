"""Classify every remaining failure of a scored configuration.

Phase 14 follow-up.

The headline metric says *how many* questions the system got wrong. This
script says *why*. Every failure in a ``results.jsonl`` is re-executed
against PostgreSQL alongside its gold query and sorted into one bucket:

    projection_only   right rows, different column set   (benchmark ambiguity)
    column_order      right rows and columns, different column order
    row_order         right rows, wrong order on an ORDER BY question
    wrong_values      same row identity, different values (bad aggregation)
    wrong_rows        genuinely different rows           (real model errors)
    schema_hallucination / syntax_error / execution_error  (detectable)

``wrong_rows`` is further tagged by comparing the *shape* of the two queries
with SQLGlot (join count, GROUP BY, LIMIT, window functions) so the report
can say whether the model missed a join or mis-filtered.

This is a **post-hoc diagnostic on the test set.** It exists to explain the
frozen number, not to change it. Any fix motivated by it — a training-data
change, a metric change — must be decided on the validation split and
re-measured as a new configuration. The headline stays what it is.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/failure_analysis.py
    env\\Scripts\\python.exe scripts/failure_analysis.py --results experiments/finetuned/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import psycopg
import sqlglot
from sqlglot import exp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_versions import add_version_argument, get as get_version
from src.sql.config import PROJECT_ROOT, ConfigError, app_config  # noqa: E402
from src.sql.executor import (  # noqa: E402
    MAX_ROWS,
    load_schema_info,
    read_only_connection,
    result_fingerprint,
)



BUCKET_ORDER = [
    "projection_only",
    "column_order",
    "row_order",
    "wrong_values",
    "wrong_rows",
    "schema_hallucination",
    "execution_error",
    "syntax_error",
]

BUCKET_MEANING = {
    "projection_only": "right rows, different column set — the question never said which columns",
    "column_order": "right rows and columns, different column order",
    "row_order": "right rows, wrong order on a question where order matters",
    "wrong_values": "same rows identified, different values — a bad computation",
    "wrong_rows": "genuinely different rows — a real logic error",
    "schema_hallucination": "referenced a table or column that does not exist",
    "execution_error": "ran but PostgreSQL rejected it (type error, ambiguity, ...)",
    "syntax_error": "did not parse",
}


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def fetch(conn: psycopg.Connection, sql: str) -> tuple[list[str], list[tuple]] | None:
    """Rows and column names, or None if the query fails."""
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            if cur.description is None:
                return [], []
            cols = [d.name for d in cur.description]
            return cols, cur.fetchmany(MAX_ROWS)
    except psycopg.Error:
        return None


def project(cols: list[str], rows: list[tuple], keep: list[str]) -> list[tuple]:
    idx = [cols.index(c) for c in keep]
    return [tuple(r[i] for i in idx) for r in rows]


# --------------------------------------------------------------------------
# SQL shape, for tagging wrong_rows
# --------------------------------------------------------------------------

def shape(sql: str) -> dict[str, Any]:
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:  # noqa: BLE001 - any parse failure is "unknown shape"
        return {"parsed": False}
    return {
        "parsed": True,
        "joins": len(list(tree.find_all(exp.Join))),
        "group_by": tree.find(exp.Group) is not None,
        "having": tree.find(exp.Having) is not None,
        "limit": tree.find(exp.Limit) is not None,
        "order_by": tree.find(exp.Order) is not None,
        "window": tree.find(exp.Window) is not None,
        "distinct": tree.find(exp.Distinct) is not None,
        "subquery": len(list(tree.find_all(exp.Subquery))) + len(list(tree.find_all(exp.CTE))),
        "null_fn": any(isinstance(n, (exp.Coalesce, exp.Nullif))
                       for n in tree.walk()),
        "tables": sorted({t.name for t in tree.find_all(exp.Table)}),
    }


def tag_wrong_rows(gold: dict, pred: dict, gold_n: int, pred_n: int) -> str:
    """One label for *how* the row sets diverged. Priority order matters."""
    if not (gold.get("parsed") and pred.get("parsed")):
        return "unparsed"
    if set(gold["tables"]) != set(pred["tables"]):
        return "wrong_tables"
    if gold["joins"] != pred["joins"]:
        return "join_structure"
    if gold["group_by"] != pred["group_by"] or gold["having"] != pred["having"]:
        return "aggregation_structure"
    if gold["window"] != pred["window"]:
        return "window_function"
    if gold["limit"] and (pred_n != gold_n or not pred["limit"]):
        return "limit_or_ranking"
    if gold["null_fn"] != pred["null_fn"]:
        return "null_handling"
    if gold["distinct"] != pred["distinct"]:
        return "distinct"
    return "filter_logic"


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def classify(conn: psycopg.Connection, row: dict, schema_columns: set[str]) -> dict:
    out = {
        "example_id": row["example_id"],
        "template_id": row.get("template_id"),
        "difficulty": row.get("difficulty"),
        "question": row.get("question"),
        "gold_sql": row["gold_sql"],
        "predicted_sql": row.get("predicted_sql") or "",
        "outcome": row.get("outcome"),
    }
    outcome = row.get("outcome")
    if outcome in ("unknown_column", "unknown_table"):
        return {**out, "bucket": "schema_hallucination", "tag": outcome}
    if outcome == "syntax_error":
        return {**out, "bucket": "syntax_error", "tag": "syntax_error"}
    if outcome != "wrong_result":
        return {**out, "bucket": "execution_error",
                "tag": (row.get("error_message") or outcome or "")[:80]}

    g = fetch(conn, row["gold_sql"])
    p = fetch(conn, out["predicted_sql"])
    if g is None or p is None:
        return {**out, "bucket": "execution_error", "tag": "re-execution failed"}
    gcols, grows = g
    pcols, prows = p
    out["gold_columns"] = gcols
    out["pred_columns"] = pcols
    out["gold_rows"] = len(grows)
    out["pred_rows"] = len(prows)

    common = [c for c in gcols if c in pcols]
    if common and len(grows) == len(prows):
        same_on_common = (result_fingerprint(project(gcols, grows, common))
                          == result_fingerprint(project(pcols, prows, common)))
    else:
        same_on_common = False

    if same_on_common:
        if set(gcols) != set(pcols):
            missing = [c for c in gcols if c not in pcols]
            extra = [c for c in pcols if c not in gcols]
            if missing and not extra:
                return {**out, "bucket": "projection_only",
                        "tag": "missing " + ", ".join(missing)}
            if extra and not missing:
                return {**out, "bucket": "projection_only",
                        "tag": "extra " + ", ".join(extra)}
            # A swap. Safe only if what went missing is a raw table column
            # the model simply chose not to return. If a *computed* column
            # (an alias like avg_days) is missing, the value the question
            # asked for was never verified - unless the model returned the
            # same values under another name, which a positional comparison
            # catches.
            if len(gcols) == len(pcols) and \
                    result_fingerprint(grows) == result_fingerprint(prows):
                return {**out, "bucket": "projection_only",
                        "tag": f"renamed {', '.join(missing)} -> {', '.join(extra)}"}
            computed = [c for c in missing if c not in schema_columns]
            if computed:
                return {**out, "bucket": "wrong_values",
                        "tag": f"computed column not returned: {', '.join(computed)}"}
            return {**out, "bucket": "projection_only",
                    "tag": f"swapped {', '.join(missing)} -> {', '.join(extra)}"}
        if gcols != pcols:
            return {**out, "bucket": "column_order", "tag": "column_order"}
        return {**out, "bucket": "row_order", "tag": "row_order"}

    gs, ps = shape(row["gold_sql"]), shape(out["predicted_sql"])
    if len(grows) == len(prows) and common:
        # Same number of rows, some shared columns, values differ.
        # Check whether the shared *identifying* columns still line up.
        key = [c for c in common if c.endswith("_id") or c in ("name", "email")]
        if key and (result_fingerprint(project(gcols, grows, key))
                    == result_fingerprint(project(pcols, prows, key))):
            return {**out, "bucket": "wrong_values",
                    "tag": tag_wrong_rows(gs, ps, len(grows), len(prows))}
    return {**out, "bucket": "wrong_rows",
            "tag": tag_wrong_rows(gs, ps, len(grows), len(prows))}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f} %" if d else "—"


def md_table(header: list[str], rows: list[list[Any]]) -> str:
    align = ["---" if i == 0 else "---:" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def write_report(results_path: Path, all_rows: list[dict], failures: list[dict],
                 config_label: str) -> None:
    n = len(all_rows)
    n_correct = sum(1 for r in all_rows if r.get("correct"))
    by_bucket = Counter(f["bucket"] for f in failures)
    tiers = ["easy", "medium", "hard", "very_hard", "enterprise"]
    tier_n = Counter(r["difficulty"] for r in all_rows)
    tier_correct = Counter(r["difficulty"] for r in all_rows if r.get("correct"))
    tier_bucket: dict[str, Counter] = defaultdict(Counter)
    for f in failures:
        tier_bucket[f["difficulty"]][f["bucket"]] += 1

    proj = by_bucket.get("projection_only", 0)
    benign = proj + by_bucket.get("column_order", 0)
    logic = by_bucket.get("wrong_rows", 0) + by_bucket.get("wrong_values", 0)
    detectable = sum(by_bucket.get(b, 0) for b in
                     ("schema_hallucination", "execution_error", "syntax_error"))

    lines: list[str] = []
    a = lines.append
    a("# Failure analysis")
    a("")
    a(f"Configuration: **{config_label}** — `{results_path.relative_to(PROJECT_ROOT).as_posix()}`")
    a("")
    a(f"{n} test questions, **{n_correct} correct ({pct(n_correct, n)})**, "
      f"{len(failures)} failures. Every failure was re-executed against PostgreSQL "
      f"next to its gold query and sorted into one bucket.")
    a("")
    a("> **This is a post-hoc diagnostic on the test set.** It explains the frozen")
    a("> number; it does not change it. Any fix it motivates must be chosen on the")
    a("> validation split and re-measured as a new configuration.")
    a("")
    a("## Where the failures are")
    a("")
    rows = []
    for b in BUCKET_ORDER:
        if by_bucket.get(b):
            rows.append([f"`{b}`", by_bucket[b], pct(by_bucket[b], len(failures)),
                         pct(by_bucket[b], n), BUCKET_MEANING[b]])
    a(md_table(["bucket", "n", "of failures", "of test set", "meaning"], rows))
    a("")
    a("Read it as three groups:")
    a("")
    a(f"* **{benign} benign** (`projection_only` + `column_order`) — the rows are right. "
      f"The question did not say which columns to return and the model chose a "
      f"different set from the gold query. {pct(benign, len(failures))} of all failures.")
    a(f"* **{logic} real logic errors** (`wrong_rows` + `wrong_values`) — the model "
      f"misunderstood the question or the schema. {pct(logic, len(failures))} of failures.")
    a(f"* **{detectable} detectable** (hallucination, execution, syntax) — the only "
      f"failures repair can see. {pct(detectable, len(failures))} of failures.")
    a("")
    a("### Row-level accuracy (diagnostic, not the headline)")
    a("")
    a("If a question does not specify columns and the model returns the right rows "
      "under a different column set, that is arguably correct. Counting "
      "`projection_only` and `column_order` as correct gives:")
    a("")
    a(md_table(["", "strict (headline)", "row-level (diagnostic)"],
               [["all", pct(n_correct, n), pct(n_correct + benign, n)]] +
               [[t, pct(tier_correct[t], tier_n[t]),
                 pct(tier_correct[t] + tier_bucket[t].get("projection_only", 0)
                     + tier_bucket[t].get("column_order", 0), tier_n[t])]
                for t in tiers if tier_n[t]]))
    a("")
    a("The strict number stays the headline: the benchmark was frozen before any "
      "of this was looked at, and loosening a metric after seeing test results is "
      "exactly the move this project refuses to make. The row-level figure is here "
      "so the gap between the two is visible and explained, not hidden.")
    a("")
    a("## By difficulty")
    a("")
    buckets_present = [b for b in BUCKET_ORDER if by_bucket.get(b)]
    a(md_table(["tier", "n", "correct"] + [f"`{b}`" for b in buckets_present],
               [[t, tier_n[t], tier_correct[t]] +
                [tier_bucket[t].get(b, 0) or "" for b in buckets_present]
                for t in tiers if tier_n[t]]))
    a("")

    # projection failures by template
    proj_rows = [f for f in failures if f["bucket"] == "projection_only"]
    if proj_rows:
        a("## `projection_only`: which templates")
        a("")
        a("Every question generated from the same template fails the same way, "
          "which is the signature of a convention mismatch rather than a model "
          "error — the model cannot know which four columns an unseen template's "
          "author chose.")
        a("")
        tpl_total = Counter(r["template_id"] for r in all_rows)
        tpl_fail = Counter(f["template_id"] for f in proj_rows)
        tpl_tag: dict[str, Counter] = defaultdict(Counter)
        tpl_q: dict[str, str] = {}
        for f in proj_rows:
            tpl_tag[f["template_id"]][f["tag"]] += 1
            tpl_q.setdefault(f["template_id"], f["question"])
        rows = []
        for t, k in tpl_fail.most_common():
            tag, _ = tpl_tag[t].most_common(1)[0]
            rows.append([f"`{t}`", f"{k} / {tpl_total[t]}", tpl_q[t], f"`{tag}`"])
        a(md_table(["template", "failed / total", "example question", "difference"], rows))
        a("")

    # wrong_rows tags
    logic_rows = [f for f in failures if f["bucket"] in ("wrong_rows", "wrong_values")]
    if logic_rows:
        a("## Real logic errors: what kind")
        a("")
        a("Tagged by comparing the shape of gold and predicted SQL with SQLGlot. "
          "The first structural difference found is the tag, in this priority: "
          "wrong tables, join count, GROUP BY / HAVING, window function, LIMIT, "
          "NULL handling, DISTINCT, and finally `filter_logic` when the shape "
          "matches and only the predicate differs.")
        a("")
        tag_counts = Counter(f["tag"] for f in logic_rows)
        tag_tier: dict[str, Counter] = defaultdict(Counter)
        for f in logic_rows:
            tag_tier[f["tag"]][f["difficulty"]] += 1
        a(md_table(["tag", "n"] + tiers,
                   [[f"`{t}`", k] + [tag_tier[t].get(d, 0) or "" for d in tiers]
                    for t, k in tag_counts.most_common()]))
        a("")
        a("By template. Like the projection failures, these cluster: a template "
          "either fails wholesale or not at all, which points at a *rule* the "
          "model does not know rather than at noise.")
        a("")
        tpl_total = Counter(r["template_id"] for r in all_rows)
        tpl_fail = Counter(f["template_id"] for f in logic_rows)
        tpl_tag: dict[str, Counter] = defaultdict(Counter)
        tpl_q: dict[str, str] = {}
        tpl_tier: dict[str, str] = {}
        for f in logic_rows:
            tpl_tag[f["template_id"]][f["tag"]] += 1
            tpl_q.setdefault(f["template_id"], f["question"])
            tpl_tier[f["template_id"]] = f["difficulty"]
        rows = []
        for t, k in tpl_fail.most_common():
            if k < 2:
                continue
            tag, _ = tpl_tag[t].most_common(1)[0]
            rows.append([f"`{t}`", tpl_tier[t], f"{k} / {tpl_total[t]}", tpl_q[t], f"`{tag}`"])
        a(md_table(["template", "tier", "failed / total", "example question", "tag"], rows))
        a("")
        a("(templates with a single failure omitted)")
        a("")

    a("## One example per bucket")
    a("")
    seen: set[str] = set()
    for f in failures:
        key = f["bucket"] + ":" + str(f.get("tag"))
        if f["bucket"] in seen:
            continue
        seen.add(f["bucket"])
        a(f"### `{f['bucket']}` — {f.get('tag')}")
        a("")
        a(f"**{f['question']}**  (`{f['example_id']}`, {f['difficulty']})")
        a("")
        a("```sql")
        a("-- gold")
        a(f["gold_sql"].strip())
        a("-- predicted")
        a(f["predicted_sql"].strip() or "(empty)")
        a("```")
        if "gold_rows" in f:
            a("")
            a(f"rows: gold {f['gold_rows']}, predicted {f['pred_rows']}; "
              f"columns: gold `{', '.join(f['gold_columns'])}`, "
              f"predicted `{', '.join(f['pred_columns'])}`")
        a("")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_version_argument(p)
    p.add_argument("--results", type=Path, default=None)
    p.add_argument("--label", default="fine-tuned + repair (configuration 5)")
    args = p.parse_args()

    version = get_version(args.version)
    global OUT_DIR, REPORT
    OUT_DIR = version.experiments_root / "failure_analysis"
    REPORT = version.experiments_root / "FAILURE_ANALYSIS.md"

    default_results = version.experiments_root / "repair" / "results.jsonl"
    results_path = (args.results or default_results).resolve()
    if results_path != default_results.resolve():
        # A one-off comparison must not overwrite the configuration-5 report.
        stem = results_path.parent.name
        OUT_DIR = version.experiments_root / f"failure_analysis_{stem}"
        REPORT = version.experiments_root / f"FAILURE_ANALYSIS_{stem}.md"
    all_rows = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines()
                if l.strip()]
    failures_in = [r for r in all_rows if not r.get("correct")]
    print(f"results  : {results_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"rows     : {len(all_rows)}  failures: {len(failures_in)}")

    try:
        cfg = app_config()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    classified: list[dict] = []
    with read_only_connection(cfg) as conn:
        schema_columns = load_schema_info(conn).all_columns
        for i, row in enumerate(failures_in, 1):
            classified.append(classify(conn, row, schema_columns))
            if i % 25 == 0:
                print(f"  classified {i}/{len(failures_in)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "failures.jsonl").open("w", encoding="utf-8") as fh:
        for c in classified:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    counts = Counter(c["bucket"] for c in classified)
    print()
    for b in BUCKET_ORDER:
        if counts.get(b):
            print(f"  {b:<22}{counts[b]:>5}")
    print(f"  {'total':<22}{len(classified):>5}")

    write_report(results_path, all_rows, classified, args.label)
    print(f"\nreport   : {REPORT.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"detail   : {(OUT_DIR / 'failures.jsonl').relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
