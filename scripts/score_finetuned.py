"""Score remotely-generated predictions from the fine-tuned model.

Phase 10, step 3 of 3.

Takes the `predictions.jsonl` produced on a GPU host and runs it through the
*same* harness that produced the frozen 10.82 % baseline. Nothing about the
scoring path differs — same extraction, same validator, same read-only
executor, same result fingerprints, same metrics. Only the weights that wrote
the SQL changed, which is the whole point.

Two details make that claim hold:

* **SQL is extracted here, not on the GPU host.** The remote side stores raw
  model output verbatim; `extract_sql` is applied locally, by the same function
  the baseline used. A second copy of that logic on the GPU host could drift
  and would quietly change the score.
* **No model is ever called.** `ReplayModel` exists only to satisfy the
  harness's interface and to record reproducibility metadata. If a prediction
  is missing for an example, it is reported as a failure rather than silently
  regenerated.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/score_finetuned.py \\
        --predictions "C:/Users/dell/Downloads/predictions.jsonl"
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
from src.evaluation.metrics import summarise  # noqa: E402
from src.evaluation.report import render_report  # noqa: E402
from src.model.base_model import (  # noqa: E402
    DEFAULT_MODEL_ID,
    GenerationResult,
    TextToSQLModel,
    extract_sql,
)
from src.model.schema_context import build_schema_context  # noqa: E402
from src.benchmark_versions import add_version_argument, get as get_version  # noqa: E402
from src.sql.config import PROJECT_ROOT, ConfigError, app_config  # noqa: E402
from src.sql.executor import load_schema_info, read_only_connection  # noqa: E402

# Resolved per --version in main(): v1 is the frozen layout, v2 lives under
# experiments/v2/ and models/finetuned_v2/.
TEST_SET = FULL_SCHEMA_DIR = RETRIEVED_DIR = ADAPTER_DIR = BASELINE_SUMMARY = None

EXPECTED_PROMPT_FP = "8288e41a496531a9"
EXPECTED_SCHEMA_FP = "d03619e711661bc5"


class ReplayModel(TextToSQLModel):
    """Replays predictions from disk. Never generates.

    A missing prediction returns a failed generation rather than calling
    anything, so a partial run scores as a partial run instead of quietly
    mixing two sources of SQL.
    """

    model_id = DEFAULT_MODEL_ID

    def __init__(self, adapter_meta: dict[str, Any] | None) -> None:
        self._meta = adapter_meta or {}
        self.misses = 0

    def generate(self, question: str, schema: str) -> GenerationResult:
        self.misses += 1
        return GenerationResult(
            sql="", raw_output="", latency_ms=0.0, ok=False,
            error="no prediction on disk for this example",
        )

    def describe(self) -> dict[str, Any]:
        cfg = self._meta.get("config", {})
        gpu = self._meta.get("gpu", {})
        return {
            "kind": "qlora_adapter_replayed",
            "model_id": cfg.get("model_id", DEFAULT_MODEL_ID),
            "model_revision": cfg.get("model_revision"),
            "provider": "kaggle-t4-local-generation",
            "fine_tuned": True,
            "adapters": [{
                "path": "models/finetuned/final_adapter",
                "method": "qlora",
                "r": cfg.get("lora_r"),
                "alpha": cfg.get("lora_alpha"),
                "dropout": cfg.get("lora_dropout"),
                "target_modules": cfg.get("lora_target_modules"),
                "trainable_params": self._meta.get("trainable_params"),
                "epochs": cfg.get("num_epochs"),
                "train_on_completion_only": cfg.get("train_on_completion_only"),
            }],
            "quantisation": {
                "load_in_4bit": cfg.get("load_in_4bit"),
                "quant_type": cfg.get("bnb_4bit_quant_type"),
                "double_quant": cfg.get("bnb_4bit_use_double_quant"),
                "compute_dtype": self._meta.get("compute_dtype"),
            },
            "generation_host": {
                "gpu": gpu.get("name"),
                "torch": gpu.get("torch"),
                "cuda": gpu.get("cuda"),
            },
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 10 fine-tuned model scoring")
    p.add_argument("--predictions", required=True,
                   help="predictions.jsonl downloaded from the GPU host")
    add_version_argument(p)
    p.add_argument("--tag", default=None, help="suffix for output files")
    p.add_argument("--allow-partial", action="store_true",
                   help="score even if predictions cover fewer than all examples")
    return p.parse_args()


def output_dir_for(header: dict, full_dir: Path, retrieved_dir: Path,
                   baseline_dir: Path | None = None) -> tuple[Path, str]:
    """Where results go, decided by what the GPU host says it ran.

    Configuration 3 and configuration 4 differ only in the schema each question
    saw, so their prediction files look almost identical. Routing them by the
    header's declared mode is what stops one silently overwriting the other.
    """
    mode = (header or {}).get("schema_mode", "full")
    # A file generated without an adapter is the base-model row of the
    # ablation; it must never land in the fine-tuned directory.
    if (header or {}).get("model_kind") == "base_4bit" and baseline_dir is not None:
        return baseline_dir, "base"
    if mode == "retrieved":
        return retrieved_dir, mode
    return full_dir, mode


def load_predictions(path: Path) -> tuple[dict[str, dict], dict]:
    """Read the remote output; returns predictions by example id, plus header."""
    header: dict = {}
    preds: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("_header"):
                header = row
                continue
            preds[row["example_id"]] = row
    return preds, header


def main() -> int:
    args = parse_args()
    version = get_version(args.version)
    prompt = version.prompt()
    TEST_SET = version.split("test")
    FULL_SCHEMA_DIR = version.experiments_root / "finetuned"
    RETRIEVED_DIR = version.experiments_root / "finetuned_retrieval"
    ADAPTER_DIR = PROJECT_ROOT / "models" / ("finetuned" if version.is_v1 else f"finetuned_{version.name}")
    BASELINE_SUMMARY = version.experiments_root / "baseline" / "summary.json"
    expected_prompt_fp = EXPECTED_PROMPT_FP if version.is_v1 else prompt.prompt_fingerprint()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"[error] predictions not found: {pred_path}", file=sys.stderr)
        return 2

    try:
        cfg = app_config()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    print("=" * 72)
    print("PHASE 10 - FINE-TUNED MODEL BENCHMARK")
    print("=" * 72)
    print(f"benchmark       : {version.name}   prompt {prompt.PROMPT_VERSION} {prompt.prompt_fingerprint()}")

    preds, header = load_predictions(pred_path)
    examples = load_examples(TEST_SET)

    print(f"test split      : {TEST_SET.relative_to(PROJECT_ROOT)}  "
          f"({len(examples):,} examples)")
    print(f"predictions     : {pred_path.name}  ({len(preds):,} rows)")

    # The remote host echoes back the fingerprints it actually used. If it
    # rendered a different prompt or schema, the comparison is void.
    OUTPUT_DIR, schema_mode = output_dir_for(header, FULL_SCHEMA_DIR, RETRIEVED_DIR,
                                             BASELINE_SUMMARY.parent)
    is_base_run = schema_mode == "base"
    if is_base_run and version.is_v1:
        print("[abort] the v1 base model is the frozen HF-inference baseline; a 4-bit "
              "Kaggle base run is only defined for v2 and later", file=sys.stderr)
        return 2
    print(f"schema mode     : {schema_mode}"
          f"{'  (ablation configuration 4)' if schema_mode == 'retrieved' else ''}")

    if header:
        got = header.get("prompt_fingerprint")
        declared = header.get("benchmark_version", "v1")
        if declared != version.name:
            print(f"[abort] predictions declare benchmark {declared!r}, but scoring "
                  f"--version {version.name}", file=sys.stderr)
            return 2
        if got != expected_prompt_fp:
            print(f"[abort] remote prompt_fingerprint = {got}, expected "
                  f"{expected_prompt_fp}.\n"
                  f"        The fine-tuned model did not see what the baseline "
                  f"saw; the comparison would be invalid.", file=sys.stderr)
            return 2
        print(f"prompt          : {got}  OK")

        # In retrieved mode there is no single schema to fingerprint: each
        # question saw a different subset, hashed collectively in the pack.
        if schema_mode != "retrieved":
            got = header.get("schema_fingerprint")
            if got != EXPECTED_SCHEMA_FP:
                print(f"[abort] remote schema_fingerprint = {got}, expected "
                      f"{EXPECTED_SCHEMA_FP}", file=sys.stderr)
                return 2
            print(f"schema          : {got}  OK")
    else:
        print("[warning] predictions carry no header; fingerprints unverified")

    missing = [e["id"] for e in examples if e["id"] not in preds]
    if missing:
        print(f"[warning] {len(missing)} examples have no prediction "
              f"(first: {missing[:3]})")
        if not args.allow_partial:
            print("[abort] refusing to score a partial run. Re-run generation, "
                  "or pass --allow-partial to score what exists.",
                  file=sys.stderr)
            return 2

    # ---- canonicalise: extract SQL here, with the baseline's own function ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"-{args.tag}" if args.tag else ""
    checkpoint = OUTPUT_DIR / f"generations{suffix}.jsonl"

    n_empty = 0
    with checkpoint.open("w", encoding="utf-8") as fh:
        for ex in examples:
            row = preds.get(ex["id"])
            if row is None:
                continue
            raw = row.get("raw_output", "")
            sql = extract_sql(raw)
            if not sql:
                n_empty += 1
            fh.write(json.dumps({
                "example_id": ex["id"],
                "sql": sql,
                "raw_output": raw,
                "latency_ms": round(float(row.get("latency_ms") or 0.0), 2),
                "ok": bool(row.get("ok", True)) and bool(sql),
                "error": row.get("error") or (None if sql else "no SQL in output"),
                "finish_reason": row.get("finish_reason"),
                "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": row.get("completion_tokens"),
                "provider": "kaggle-t4-local-generation",
                "attempts": 1,
                "fatal": False,
            }, ensure_ascii=False) + "\n")

    print(f"extracted SQL   : {len(preds) - n_empty:,} of {len(preds):,} "
          f"outputs contained a statement")
    print(f"generations     : {checkpoint.relative_to(PROJECT_ROOT)}")

    adapter_meta = None
    meta_path = ADAPTER_DIR / "training_config.json"
    if meta_path.exists():
        adapter_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        print(f"[warning] {meta_path} missing; run metadata will be thin")

    model = ReplayModel(None if is_base_run else adapter_meta)
    if is_base_run:
        model.describe = lambda: {  # type: ignore[method-assign]
            "kind": "base_model_4bit_replayed", "model_id": DEFAULT_MODEL_ID,
            "model_revision": header.get("revision"), "fine_tuned": False,
            "provider": "kaggle-t4-local-generation",
            "quantisation": {"load_in_4bit": True, "quant_type": "nf4",
                             "double_quant": True, "compute_dtype": "float16"},
            "note": ("Base model without any adapter, generated on the same "
                     "hardware and decoding as the fine-tuned run. Unlike the v1 "
                     "baseline (HF inference, full precision) this is a 4-bit load."),
            "generation_host": {"gpu": header.get("gpu")}}

    with read_only_connection(cfg) as conn:
        schema_text, schema_fp = build_schema_context(conn)
        schema_info = load_schema_info(conn)

        if schema_fp != EXPECTED_SCHEMA_FP:
            print(f"[abort] local schema fingerprint {schema_fp} != "
                  f"{EXPECTED_SCHEMA_FP}", file=sys.stderr)
            return 2

        metadata = build_run_metadata(
            model, TEST_SET, conn, schema_fp, schema_text, schema_info.tables,
            prompt=prompt,
        )
        if is_base_run:
            metadata["schema_context"]["note"] = (
                f"Benchmark {version.name} ablation configuration 1: base model, "
                "full schema, no retrieval, no repair, 4-bit on a Kaggle T4.")
        elif schema_mode == "retrieved":
            metadata["schema_context"] = {
                "mode": "retrieved_keyword_v1",
                "fingerprint": header.get("schema_fingerprint"),
                "note": (
                    "Phase 14 ablation configuration 4: fine-tuned model, "
                    "retrieved schema subset, no repair. The adapter was "
                    "trained exclusively on full-schema prompts, so this "
                    "configuration is deliberately out of distribution for it."
                ),
            }
        else:
            metadata["schema_context"]["note"] = (
                "Phase 10 ablation configuration 3: fine-tuned model, full "
                "schema, no retrieval, no repair. Matches the Phase 4 baseline "
                "exactly except for the weights."
            )
        print(f"data fingerprint: {metadata['database']['data_fingerprint']}")

        print("\nRunning gold queries and scoring predictions...")

        def progress(done: int, total: int) -> None:
            if done % 50 == 0 or done == total:
                print(f"  {done:>4}/{total}", flush=True)

        scored = [e for e in examples if e["id"] in preds]
        results, drift = run_benchmark(
            model, scored, conn, schema_text, schema_info,
            workers=1, progress=progress,
            checkpoint_path=checkpoint, resume=True,
        )

    if model.misses:
        print(f"[warning] harness asked to generate {model.misses} times; "
              f"those score as failures")

    summary = summarise(results)
    summary["run_metadata"] = metadata
    if drift:
        summary["gold_fingerprint_drift"] = drift
    if header:
        summary["generation_host"] = {
            k: v for k, v in header.items() if k != "_header"
        }

    results_path = OUTPUT_DIR / f"results{suffix}.jsonl"
    summary_path = OUTPUT_DIR / f"summary{suffix}.json"
    with results_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.tag:
        (OUTPUT_DIR / ("BASELINE_REPORT.md" if is_base_run else "FINETUNED_REPORT.md")).write_text(
            render_report(summary, metadata, results, drift=drift),
            encoding="utf-8")

    # ---- headline ----------------------------------------------------------
    t = summary["totals"]
    e = summary["error_rates"]
    pa = summary["projection_analysis"]

    print("\n" + "=" * 72)
    print(f"EXECUTION ACCURACY   {t['execution_accuracy_pct']:.2f} %  "
          f"({t['correct']}/{t['total_examples']})")
    print(f"executable SQL       {t['executable_sql_pct']:.2f} %")
    print(f"schema hallucination {e['schema_hallucination_pct']:.2f} %")
    print(f"syntax errors        {e['syntax_error_pct']:.2f} %")
    print(f"projection-tolerant  {pa['projection_tolerant_accuracy_pct']:.2f} %")
    print("-" * 72)
    print("outcomes:")
    for name, count in summary["outcomes"].items():
        print(f"  {name:<22}{count:>5}")

    # ---- head to head ------------------------------------------------------
    if BASELINE_SUMMARY.exists() and not is_base_run:
        base = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
        bt, be = base["totals"], base["error_rates"]
        bpa = base["projection_analysis"]
        rows = [
            ("strict execution accuracy %", bt["execution_accuracy_pct"],
             t["execution_accuracy_pct"], True),
            ("projection-tolerant %", bpa["projection_tolerant_accuracy_pct"],
             pa["projection_tolerant_accuracy_pct"], True),
            ("executable SQL %", bt["executable_sql_pct"],
             t["executable_sql_pct"], True),
            ("schema hallucination %", be["schema_hallucination_pct"],
             e["schema_hallucination_pct"], False),
            ("syntax error %", be["syntax_error_pct"],
             e["syntax_error_pct"], False),
        ]
        print("-" * 72)
        print(f"{'metric':<30}{'base':>9}{'fine-tuned':>13}{'delta':>10}")
        print("-" * 72)
        for name, b, f, higher_better in rows:
            d = f - b
            arrow = ("better" if (d > 0) == higher_better else "worse") \
                if abs(d) >= 0.005 else "same"
            print(f"{name:<30}{b:>9.2f}{f:>13.2f}{d:>+9.2f}  {arrow}")
        print("-" * 72)
        delta = t["execution_accuracy_pct"] - bt["execution_accuracy_pct"]
        verdict = ("FINE-TUNING HELPED" if delta > 0 else
                   "FINE-TUNING DID NOT HELP" if delta < 0 else
                   "NO MEASURABLE CHANGE")
        print(f"VERDICT: {verdict}  ({delta:+.2f} pp strict execution accuracy)")
    else:
        print(f"[warning] {BASELINE_SUMMARY} missing; no comparison shown")

    if drift:
        print(f"\n[warning] {len(drift)} gold fingerprint drift issue(s)")
        for line in drift[:3]:
            print(f"  {line}")
    print("=" * 72)
    print(f"\nwritten to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
