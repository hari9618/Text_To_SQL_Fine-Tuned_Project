"""Export the failing queries that Phase 11 will try to repair.

Phase 11, step 1 of 3. Same split as Phase 10: the GPU host regenerates, the
laptop decides whether the regeneration is any good.

    this script  ->  repairpack.zip  ->  Kaggle notebook  ->  repairs.jsonl
                                                                   |
                                              scripts/score_repair.py
                                                                   |
                                                    ablation configuration 5

What travels is what production would have: the question, the query that
failed, and the error the database returned. **No gold SQL, no gold result.**
A repair loop that needs the answer to fix a query is not a repair loop.

Failures are re-derived here by executing each prediction again, rather than
read out of the Phase 10 results file. The error text must be exactly what the
repair prompt will show the model, and re-deriving it proves the database still
produces that error.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/export_repair_pack.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_versions import add_version_argument, get as get_version  # noqa: E402
from src.model.repair_prompt import (  # noqa: E402
    REPAIR_PROMPT_VERSION,
    repair_prompt_fingerprint,
)
from src.model.schema_context import build_schema_context  # noqa: E402
from src.sql.config import PROJECT_ROOT, ConfigError, app_config  # noqa: E402
from src.sql.executor import load_schema_info, read_only_connection  # noqa: E402
from src.sql.repair import diagnose  # noqa: E402

EXPECTED_SCHEMA_FP = "d03619e711661bc5"

PROMPT_SRC = PROJECT_ROOT / "src" / "model" / "repair_prompt.py"


def to_beat(version) -> dict:
    """The fine-tuned numbers this repair run has to improve on. Repair is
    measured against configuration 3 of the *same* benchmark version, never
    against another version's."""
    summary = version.experiments_root / "finetuned" / "summary.json"
    if not summary.exists():
        return {"note": "configuration 3 not yet measured on this version"}
    s = json.loads(summary.read_text(encoding="utf-8"))
    return {"executable_sql_pct": s["totals"]["executable_sql_pct"],
            "strict_execution_accuracy_pct": s["totals"]["execution_accuracy_pct"],
            "schema_hallucination_pct": s["error_rates"]["schema_hallucination_pct"]}


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 11 repair pack export")
    add_version_argument(p)
    p.add_argument("--results", default=None,
                   help="scored results to take failures from "
                        "(default: this version's configuration 3)")
    p.add_argument("--zip-to", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    version = get_version(args.version)
    suffix = "" if version.is_v1 else f"-{version.name}"
    out_dir = version.experiments_root / "repair" / "repairpack"
    results_path = Path(args.results or
                        version.experiments_root / "finetuned" / "results.jsonl")
    zip_to = Path(args.zip_to or
                  Path.home() / "Downloads" / f"text2sql-repairpack{suffix}.zip")
    if not results_path.exists():
        print(f"[error] results not found: {results_path}", file=sys.stderr)
        return 2

    try:
        cfg = app_config()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    print("=" * 72)
    print(f"PHASE 11 - EXPORT REPAIR PACK  (benchmark {version.name})")
    print("=" * 72)

    fp = repair_prompt_fingerprint()
    print(f"repair prompt   : {REPAIR_PROMPT_VERSION}  {fp}")

    rows = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    print(f"scored results  : {results_path.relative_to(PROJECT_ROOT)} "
          f"({len(rows):,} examples)")

    with read_only_connection(cfg) as conn:
        schema_text, schema_fp = build_schema_context(conn)
        if schema_fp != EXPECTED_SCHEMA_FP:
            print(f"[abort] schema fingerprint drifted: {schema_fp}", file=sys.stderr)
            return 2
        schema_info = load_schema_info(conn)
        print(f"schema          : {schema_fp}  OK")

        failures = []
        confirmed = 0
        for r in rows:
            sql = r.get("predicted_sql") or ""
            d = diagnose(sql, conn, schema_info)
            if d.ok:
                continue
            confirmed += 1
            failures.append({
                "id": r["example_id"],
                "question": r["question"],
                "failed_sql": sql,
                "error": d.error,
                "stage": d.stage,
            })

    # Every failure the scorer counted should reproduce, and nothing else.
    scored_bad = {r["example_id"] for r in rows
                  if r["outcome"] not in ("correct", "wrong_result")}
    found = {f["id"] for f in failures}
    if found != scored_bad:
        print(f"[warning] re-derived failures differ from the scored set: "
              f"only-now={sorted(found - scored_bad)} "
              f"only-then={sorted(scored_bad - found)}")

    print(f"failures        : {len(failures)} of {len(rows)} "
          f"({100 * len(failures) / max(len(rows), 1):.2f} %)")
    by_stage: dict[str, int] = {}
    for f in failures:
        by_stage[f["stage"]] = by_stage.get(f["stage"], 0) + 1
    for stage, n in sorted(by_stage.items()):
        print(f"  {stage:<12}{n:>4}")

    if not failures:
        print("\nnothing to repair - every prediction executes.")
        return 0

    leaked = {k for f in failures for k in f} - {
        "id", "question", "failed_sql", "error", "stage"}
    assert not leaked, f"repair pack would leak: {sorted(leaked)}"
    assert not any("gold" in k for f in failures for k in f)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    blob = "\n".join(json.dumps(f, ensure_ascii=False, sort_keys=True)
                     for f in failures)
    (out_dir / "repair_inputs.jsonl").write_text(blob + "\n", encoding="utf-8")
    shutil.copyfile(PROMPT_SRC, out_dir / "repair_prompt_module.py")

    manifest = {
        "phase": 11,
        "benchmark_version": version.name,
        "purpose": "failing queries for remote repair; scoring happens locally",
        "repair_prompt": {"version": REPAIR_PROMPT_VERSION, "fingerprint": fp},
        "schema": {"fingerprint": schema_fp},
        "failures": {
            "count": len(failures),
            "fingerprint": sha16(blob),
            "by_stage": by_stage,
            "source": str(results_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "gold_sql_included": False,
        },
        "to_beat": to_beat(version),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")

    zip_path = zip_to
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out_dir.iterdir()):
            zf.write(f, f.name)

    print("-" * 72)
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.stat().st_size / 1024:8.1f} KB  {f.relative_to(PROJECT_ROOT)}")
    print("-" * 72)
    print(f"upload this: {zip_path}  ({zip_path.stat().st_size / 1024:.0f} KB)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
