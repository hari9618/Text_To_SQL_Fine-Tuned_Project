"""Run the Phase 4 base-model benchmark.

Evaluation only. No fine-tuning, no LoRA adapters, no schema retrieval, no SQL
repair. The test split is read, never modified.

Usage (from the project root):

    # harness self-test - no model calls, no cost
    env\\Scripts\\python.exe scripts/run_baseline.py --oracle --limit 20

    # smoke test against the real model
    env\\Scripts\\python.exe scripts/run_baseline.py --limit 5

    # full test set
    env\\Scripts\\python.exe scripts/run_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.baseline import (  # noqa: E402
    build_run_metadata,
    load_examples,
    run_benchmark,
    stride_sample,
)
from src.evaluation.metrics import summarise  # noqa: E402
from src.evaluation.report import render_report  # noqa: E402
from src.model.base_model import (  # noqa: E402
    DEFAULT_MODEL_ID,
    HFInferenceModel,
    InferenceParams,
    OracleModel,
)
from src.model.schema_context import build_schema_context  # noqa: E402
from src.benchmark_versions import add_version_argument, get as get_version  # noqa: E402
from src.sql.config import PROJECT_ROOT, ConfigError, app_config  # noqa: E402
from src.sql.executor import load_schema_info, read_only_connection  # noqa: E402



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 base-model benchmark")
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N examples (in file order)")
    p.add_argument("--sample", type=int, default=None,
                   help="evaluate N examples spread across the split "
                        "(preferred for smoke tests: the split is ordered by "
                        "template, so --limit returns near-duplicates)")
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--provider", default="auto",
                   help="HF inference provider, or 'auto' to let HF route")
    p.add_argument("--workers", type=int, default=4,
                   help="concurrent generation requests")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--oracle", action="store_true",
                   help="self-test the harness with gold SQL; makes no API calls")
    p.add_argument("--retrieval", choices=["none", "keyword"], default="none",
                   help="schema mode: 'none' shows the full schema (Phase 5 "
                        "baseline), 'keyword' shows a retrieved subset")
    p.add_argument("--retrieval-k", type=int, default=4,
                   help="max seed tables (tuned on the validation split)")
    p.add_argument("--retrieval-expand", type=int, default=2,
                   help="FK neighbour expansion (tuned on the validation split)")
    p.add_argument("--only-checkpointed", action="store_true",
                   help="score only examples already generated on disk; makes "
                        "no API calls. Use to salvage a run cut short by "
                        "provider limits.")
    p.add_argument("--resume", action="store_true",
                   help="reuse generations already checkpointed on disk")
    add_version_argument(p)
    p.add_argument("--tag", default=None,
                   help="suffix for output files, e.g. 'smoke'")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    version = get_version(args.version)
    prompt = version.prompt()
    TEST_SET = version.split("test")
    BASELINE_DIR = version.experiments_root / "baseline"
    RETRIEVAL_DIR = version.experiments_root / "retrieval"

    if not TEST_SET.exists():
        print(f"[error] test split not found: {TEST_SET}", file=sys.stderr)
        return 2

    try:
        cfg = app_config()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    examples = load_examples(TEST_SET, args.limit)
    if args.sample:
        examples = stride_sample(load_examples(TEST_SET), args.sample)
    print("=" * 72)
    print("PHASE 4 — BASE MODEL BENCHMARK")
    print("=" * 72)
    print(f"benchmark       : {version.name}   prompt {prompt.PROMPT_VERSION} {prompt.prompt_fingerprint()}")
    print(f"test split      : {TEST_SET.relative_to(PROJECT_ROOT)}")
    print(f"examples        : {len(examples):,}"
          f"{f' (limited from full split)' if args.limit else ''}")

    # The model is built before the database connection so a missing HF_TOKEN
    # fails immediately rather than after minutes of gold execution.
    if args.oracle:
        model = OracleModel({e["question"]: e["sql"] for e in examples})
        print("model           : ORACLE (harness self-test, no API calls)")
    else:
        try:
            model = HFInferenceModel(
                model_id=args.model,
                provider=args.provider,
                params=InferenceParams(
                    temperature=args.temperature, max_tokens=args.max_tokens
                ),
                prompt=prompt,
            )
        except RuntimeError as exc:
            print(f"\n[error] {exc}", file=sys.stderr)
            return 2
        print(f"model           : {args.model} (provider={args.provider})")
        print(f"inference       : temperature={args.temperature}, "
              f"max_tokens={args.max_tokens}, thinking disabled")

    OUTPUT_DIR = BASELINE_DIR if args.retrieval == "none" else RETRIEVAL_DIR

    with read_only_connection(cfg) as conn:
        schema_text, schema_fp = build_schema_context(conn)
        schema_info = load_schema_info(conn)

        schema_for = None
        retriever = None
        if args.retrieval == "keyword":
            from src.retrieval.keyword import KeywordRetriever
            from src.retrieval.schema_index import build_index

            retriever = KeywordRetriever(
                build_index(conn), max_tables=args.retrieval_k,
                expand_neighbours=args.retrieval_expand,
            )
            cache: dict[str, tuple[str, dict]] = {}

            def schema_for(example):
                key = example["question"]
                if key not in cache:
                    r = retriever.retrieve(key)
                    gold = set(example.get("referenced_tables") or [])
                    got = set(r.tables)
                    cache[key] = (r.schema_text, {
                        "tables": r.tables,
                        "bridge_tables": r.bridge_tables,
                        "n_tables": len(r.tables),
                        "schema_chars": len(r.schema_text),
                        "gold_tables": sorted(gold),
                        "gold_total": len(gold),
                        "gold_found": len(gold & got),
                        "missing_tables": sorted(gold - got),
                        "complete": gold.issubset(got),
                        "matched_values": r.matched_values,
                    })
                return cache[key]

            print(f"schema context  : RETRIEVED subset "
                  f"(k={args.retrieval_k}, expand={args.retrieval_expand}) "
                  f"from {len(schema_info.tables)} tables")
        else:
            print(f"schema context  : full schema, {len(schema_info.tables)} tables, "
                  f"{len(schema_text):,} chars (hash {schema_fp})")

        metadata = build_run_metadata(
            model, TEST_SET, conn, schema_fp, schema_text, schema_info.tables,
            prompt=prompt,
        )
        print(f"data fingerprint: {metadata['database']['data_fingerprint']}")
        print(f"DATA_AS_OF      : {metadata['database']['data_as_of']}")

        print("\nRunning gold queries and generating predictions...")

        def progress(done: int, total: int) -> None:
            if done % 10 == 0 or done == total:
                print(f"  {done:>4}/{total}", flush=True)

        checkpoint = OUTPUT_DIR / f"generations{f'-{args.tag}' if args.tag else ''}.jsonl"
        if args.only_checkpointed and checkpoint.exists():
            import json as _json
            done = {
                _json.loads(l)["example_id"]
                for l in checkpoint.read_text(encoding="utf-8").splitlines() if l.strip()
            }
            examples = [e for e in examples if e["id"] in done]
            print(f"  restricted to {len(examples):,} already-generated examples")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if not args.resume and checkpoint.exists():
            checkpoint.unlink()

        results, drift = run_benchmark(
            model, examples, conn, schema_text, schema_info,
            workers=1 if args.oracle else args.workers,
            progress=progress,
            checkpoint_path=None if args.oracle else checkpoint,
            resume=args.resume,
            schema_for=schema_for,
        )

    summary = summarise(results)
    if retriever is not None:
        metadata["schema_context"] = {
            "mode": f"retrieved_{retriever.name}",
            **retriever.describe(),
        }
    summary["run_metadata"] = metadata
    if drift:
        summary["gold_fingerprint_drift"] = drift

    suffix = f"-{args.tag}" if args.tag else ""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / f"results{suffix}.jsonl"
    summary_path = OUTPUT_DIR / f"summary{suffix}.json"

    with results_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # The full run owns BASELINE_REPORT.md; smoke and oracle runs write their
    # own tagged files so they cannot overwrite the headline result.
    if not args.tag:
        report = render_report(summary, metadata, results, drift=drift)
        report_name = ("BASELINE_REPORT.md" if args.retrieval == "none"
                       else "RUN_REPORT.md")
        (OUTPUT_DIR / report_name).write_text(report, encoding="utf-8")

    t = summary["totals"]
    e = summary["error_rates"]
    print("\n" + "=" * 72)
    print(f"EXECUTION ACCURACY   {t['execution_accuracy_pct']:.2f} %  "
          f"({t['correct']}/{t['total_examples']})")
    print(f"executable SQL       {t['executable_sql_pct']:.2f} %")
    print(f"ran but wrong        {t['wrong_result_pct']:.2f} %")
    pa = summary["projection_analysis"]
    print(f"schema hallucination {e['schema_hallucination_pct']:.2f} %")
    print("-" * 72)
    print("wrong_result breakdown:")
    print(f"  same rows, different columns  {pa['projection_mismatch_only']:>5}"
          f"  ({pa['projection_mismatch_only_pct']:.2f} %)  <- benchmark artefact")
    print(f"  genuinely different rows      {pa['genuinely_different_rows']:>5}"
          f"  ({pa['genuinely_different_rows_pct']:.2f} %)  <- real SQL error")
    print(f"  projection-tolerant accuracy  "
          f"{pa['projection_tolerant_accuracy_pct']:.2f} %")
    print(f"syntax errors        {e['syntax_error_pct']:.2f} %")
    print("-" * 72)
    print("outcomes:")
    for name, count in summary["outcomes"].items():
        print(f"  {name:<22}{count:>5}")
    print("-" * 72)
    rq = summary.get("retrieval")
    if rq:
        print("retrieval:")
        print(f"  table recall                {rq['table_recall_pct']:.2f} %")
        print(f"  complete retrieval          {rq['complete_retrieval_pct']:.2f} %")
        print(f"  mean tables shown           {rq['mean_tables_shown']:.2f}")
        print(f"  mean schema chars           {rq['mean_schema_chars']:,.0f}")
        print(f"  cited a table never shown   {rq['referenced_a_table_never_shown']}")
        print(f"  accuracy | retrieval OK     {rq['accuracy_when_retrieval_complete_pct']:.2f} %")
        print(f"  accuracy | retrieval missed {rq['accuracy_when_retrieval_incomplete_pct']:.2f} %")
        print("-" * 72)

    lat = summary["latency"]
    print(f"generation latency   mean {lat['generation']['mean_ms']:.0f} ms | "
          f"median {lat['generation']['median_ms']:.0f} ms")
    print(f"total latency        mean {lat['total']['mean_ms']:.0f} ms")
    if drift:
        print(f"\n[warning] {len(drift)} gold fingerprint drift issue(s)")
        for line in drift[:3]:
            print(f"  {line}")
    print("=" * 72)
    print(f"\nwritten to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
