"""Score the repair loop: ablation configuration 5.

Phase 11, step 3 of 3.

Folds repaired SQL into the Phase 10 predictions and rescores the whole 453
through the same harness, so configuration 5 (fine-tuned + full schema +
repair) is directly comparable to configuration 3 (fine-tuned, no repair) and
to the frozen baseline.

Two rates are reported, and they are not the same thing:

* **repair success** - the query now executes. This is what the loop can
  actually detect and act on in production.
* **repair correctness** - the query now returns the gold rows. Strictly
  harder, and the only one that moves execution accuracy.

A repair that turns a crash into a confident wrong answer counts as a success
on the first measure and a failure on the second. Reporting only the first
would overstate what repair achieves.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/score_repair.py \\
        --repairs "C:/Users/dell/Downloads/repairs.jsonl"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.baseline import (  # noqa: E402
    build_run_metadata,
    load_examples,
    run_benchmark,
)
from src.benchmark_versions import add_version_argument, get as get_version  # noqa: E402
from src.evaluation.metrics import summarise  # noqa: E402
from src.evaluation.report import render_report  # noqa: E402
from src.model.base_model import extract_sql  # noqa: E402
from src.model.repair_prompt import (  # noqa: E402
    REPAIR_PROMPT_VERSION,
    repair_prompt_fingerprint,
)
from src.model.schema_context import build_schema_context  # noqa: E402
from src.sql.config import PROJECT_ROOT, ConfigError, app_config  # noqa: E402
from src.sql.executor import load_schema_info, read_only_connection  # noqa: E402
from src.sql.repair import diagnose  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from score_finetuned import ReplayModel  # noqa: E402  (same replay semantics)

EXPECTED_SCHEMA_FP = "d03619e711661bc5"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 11 repair scoring")
    add_version_argument(p)
    p.add_argument("--repairs", required=True,
                   help="repairs.jsonl downloaded from the GPU host")
    p.add_argument("--generations", default=None,
                   help="the Phase 10 generations to fold repairs into "
                        "(default: this version's configuration 3)")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def load_jsonl(path: Path) -> tuple[list[dict], dict]:
    header: dict = {}
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("_header"):
            header = row
        else:
            rows.append(row)
    return rows, header


def main() -> int:
    args = parse_args()
    version = get_version(args.version)
    TEST_SET = version.split("test")
    FINETUNED_DIR = version.experiments_root / "finetuned"
    REPAIR_DIR = version.experiments_root / "repair"
    ADAPTER_DIR = PROJECT_ROOT / "models" / (
        "finetuned" if version.is_v1 else f"finetuned_{version.name}")
    BASELINE_SUMMARY = version.experiments_root / "baseline" / "summary.json"

    repairs_path = Path(args.repairs)
    generations_path = Path(args.generations or FINETUNED_DIR / "generations.jsonl")
    for path in (repairs_path, generations_path):
        if not path.exists():
            print(f"[error] not found: {path}", file=sys.stderr)
            return 2

    try:
        cfg = app_config()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    print("=" * 72)
    print(f"PHASE 11 - SQL REPAIR (ablation configuration 5, benchmark {version.name})")
    print("=" * 72)

    repair_rows, header = load_jsonl(repairs_path)
    repairs = {r["example_id"]: r for r in repair_rows}
    base_gens, _ = load_jsonl(generations_path)

    print(f"phase 10 generations : {len(base_gens):,}")
    print(f"repairs              : {len(repairs):,}")

    if header:
        got = header.get("repair_prompt_fingerprint")
        want = repair_prompt_fingerprint()
        if got != want:
            print(f"[abort] remote repair prompt {got} != local {want}",
                  file=sys.stderr)
            return 2
        if header.get("schema_fingerprint") != EXPECTED_SCHEMA_FP:
            print(f"[abort] remote schema {header.get('schema_fingerprint')} "
                  f"!= {EXPECTED_SCHEMA_FP}", file=sys.stderr)
            return 2
        print(f"repair prompt        : {REPAIR_PROMPT_VERSION}  {got}  OK")
        print(f"schema               : {header.get('schema_fingerprint')}  OK")
    else:
        print("[warning] repairs carry no header; fingerprints unverified")

    examples = load_examples(TEST_SET)
    by_id = {e["id"]: e for e in examples}

    REPAIR_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"-{args.tag}" if args.tag else ""

    # ---- diagnose before and after, then fold in ---------------------------
    audit: list[dict[str, Any]] = []
    merged: list[dict] = []
    n_unchanged = 0

    with read_only_connection(cfg) as conn:
        schema_text, schema_fp = build_schema_context(conn)
        if schema_fp != EXPECTED_SCHEMA_FP:
            print(f"[abort] local schema {schema_fp} != {EXPECTED_SCHEMA_FP}",
                  file=sys.stderr)
            return 2
        schema_info = load_schema_info(conn)

        for gen in base_gens:
            eid = gen["example_id"]
            rep = repairs.get(eid)
            if rep is None:
                merged.append(gen)
                continue

            before_sql = gen.get("sql", "")
            after_sql = extract_sql(rep.get("raw_output", ""))
            before = diagnose(before_sql, conn, schema_info)
            after = diagnose(after_sql, conn, schema_info)

            if after_sql.strip() == before_sql.strip():
                n_unchanged += 1

            audit.append({
                "example_id": eid,
                "question": by_id[eid]["question"] if eid in by_id else None,
                "before_sql": before_sql,
                "before_error": before.error,
                "after_sql": after_sql,
                "after_error": after.error,
                "now_executes": after.ok,
                "was_broken": not before.ok,
                "unchanged": after_sql.strip() == before_sql.strip(),
                "repair_latency_ms": rep.get("latency_ms"),
            })

            merged.append({
                **gen,
                "sql": after_sql,
                "raw_output": rep.get("raw_output", ""),
                "latency_ms": round(
                    float(gen.get("latency_ms") or 0.0)
                    + float(rep.get("latency_ms") or 0.0), 2),
                "ok": bool(after_sql),
                "error": None if after_sql else "no SQL in repair output",
                "repaired": True,
                "prompt_tokens": rep.get("prompt_tokens"),
                "completion_tokens": rep.get("completion_tokens"),
            })

        attempted = [a for a in audit if a["was_broken"]]
        fixed = [a for a in attempted if a["now_executes"]]

        print(f"repair attempted     : {len(attempted)}")
        print(f"now executes         : {len(fixed)}")
        if n_unchanged:
            print(f"model returned the same SQL: {n_unchanged}")

        checkpoint = REPAIR_DIR / f"generations{suffix}.jsonl"
        with checkpoint.open("w", encoding="utf-8") as fh:
            for row in merged:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        adapter_meta = None
        meta_path = ADAPTER_DIR / "training_config.json"
        if meta_path.exists():
            adapter_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model = ReplayModel(adapter_meta)

        metadata = build_run_metadata(
            model, TEST_SET, conn, schema_fp, schema_text, schema_info.tables)
        metadata["model"]["repair"] = {
            "enabled": True,
            "prompt_version": REPAIR_PROMPT_VERSION,
            "prompt_fingerprint": repair_prompt_fingerprint(),
            "max_repairs": 1,
            "triggered_on": "static validation failure or execution error",
            "note": ("repair never sees gold SQL or gold results; it is driven "
                     "only by the error the database returned"),
        }
        metadata["schema_context"]["note"] = (
            "Phase 11 ablation configuration 5: fine-tuned model, full schema, "
            "one repair attempt on detectable failures.")

        print("\nRescoring all 453 with repairs folded in...")

        def progress(done: int, total: int) -> None:
            if done % 100 == 0 or done == total:
                print(f"  {done:>4}/{total}", flush=True)

        results, drift = run_benchmark(
            model, examples, conn, schema_text, schema_info,
            workers=1, progress=progress,
            checkpoint_path=checkpoint, resume=True)

    summary = summarise(results)
    summary["run_metadata"] = metadata
    if drift:
        summary["gold_fingerprint_drift"] = drift

    # ---- repair-specific numbers ------------------------------------------
    by_result = {r.example_id: r for r in results}
    now_correct = [a for a in fixed if getattr(by_result.get(a["example_id"]),
                                               "correct", False)]
    summary["repair"] = {
        "attempted": len(attempted),
        "now_executes": len(fixed),
        "success_rate_pct": round(100 * len(fixed) / max(len(attempted), 1), 2),
        "now_correct": len(now_correct),
        "correctness_rate_pct": round(
            100 * len(now_correct) / max(len(attempted), 1), 2),
        "returned_identical_sql": n_unchanged,
        "mean_repair_latency_ms": round(
            sum(a["repair_latency_ms"] or 0 for a in attempted)
            / max(len(attempted), 1), 2),
    }

    (REPAIR_DIR / f"repair_audit{suffix}.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in audit) + "\n",
        encoding="utf-8")
    with (REPAIR_DIR / f"results{suffix}.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")
    (REPAIR_DIR / f"summary{suffix}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    if not args.tag:
        (REPAIR_DIR / "REPAIR_REPORT.md").write_text(
            render_report(summary, metadata, results, drift=drift),
            encoding="utf-8")

    # ---- headline ----------------------------------------------------------
    t, e = summary["totals"], summary["error_rates"]
    rp = summary["repair"]
    print("\n" + "=" * 72)
    print(f"EXECUTION ACCURACY   {t['execution_accuracy_pct']:.2f} %  "
          f"({t['correct']}/{t['total_examples']})")
    print(f"executable SQL       {t['executable_sql_pct']:.2f} %")
    print(f"schema hallucination {e['schema_hallucination_pct']:.2f} %")
    print("-" * 72)
    print("repair:")
    print(f"  attempted                  {rp['attempted']:>5}")
    print(f"  now executes               {rp['now_executes']:>5}"
          f"  ({rp['success_rate_pct']:.1f} % success)")
    print(f"  now returns the gold rows  {rp['now_correct']:>5}"
          f"  ({rp['correctness_rate_pct']:.1f} %)")
    print(f"  model returned same SQL    {rp['returned_identical_sql']:>5}")
    print(f"  mean repair latency        {rp['mean_repair_latency_ms']:>7.0f} ms")

    # ---- three-way comparison ---------------------------------------------
    rows = []
    for label, path in (("base", BASELINE_SUMMARY),
                        ("fine-tuned", FINETUNED_DIR / "summary.json")):
        if path.exists():
            rows.append((label, json.loads(path.read_text(encoding="utf-8"))))
    rows.append(("+ repair", summary))

    print("-" * 72)
    print(f"{'metric':<28}" + "".join(f"{lbl:>14}" for lbl, _ in rows))
    print("-" * 72)
    for name, get in (
        ("strict execution acc %", lambda s: s["totals"]["execution_accuracy_pct"]),
        ("executable SQL %", lambda s: s["totals"]["executable_sql_pct"]),
        ("schema hallucination %", lambda s: s["error_rates"]["schema_hallucination_pct"]),
        ("syntax error %", lambda s: s["error_rates"]["syntax_error_pct"]),
    ):
        print(f"{name:<28}" + "".join(f"{get(s):>14.2f}" for _, s in rows))
    print("=" * 72)
    print(f"\nwritten to {REPAIR_DIR.relative_to(PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
