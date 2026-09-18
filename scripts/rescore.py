"""Re-score existing predictions against a different benchmark version.

Phase 15.

Every measured configuration checkpointed its generations
(``experiments/<config>/generations.jsonl``). Those are the model's outputs
for the 453 test questions and do not depend on the gold SQL, so they can be
scored against benchmark v2's gold without generating anything. That gives
the v1 models a v2 number for free and, more importantly, shows how much of
each configuration's v1 score was the benchmark's own column conventions.

What this is and is not:

* The predictions were generated with **prompt v1** against the **v1 question
  text**. 445 of 453 v2 questions are byte-identical; the 8 reworded ``x14``
  paraphrases are scored against the answer the model gave to the old
  wording. Both facts are recorded in the summary.
* It is a *re-scoring*, not a new run. A v2-prompted, v2-trained model is a
  separate configuration and gets its own generation on the GPU host.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/rescore.py --version v2 --all
    env\\Scripts\\python.exe scripts/rescore.py --version v2 --config baseline
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_versions import add_version_argument, get as get_version  # noqa: E402
from src.evaluation.baseline import (  # noqa: E402
    build_run_metadata,
    load_examples,
    run_benchmark,
)
from src.evaluation.metrics import summarise  # noqa: E402
from src.model.base_model import (  # noqa: E402
    DEFAULT_MODEL_ID,
    GenerationResult,
    TextToSQLModel,
)
from src.model.schema_context import build_schema_context  # noqa: E402
from src.sql.config import PROJECT_ROOT, ConfigError, app_config  # noqa: E402
from src.sql.executor import load_schema_info, read_only_connection  # noqa: E402

# Configuration name -> (v1 experiment directory, human label)
CONFIGS: dict[str, tuple[str, str]] = {
    "baseline": ("baseline", "1 · base model"),
    "retrieval": ("retrieval", "2 · base + retrieval"),
    "finetuned": ("finetuned", "3 · fine-tuned"),
    "finetuned_retrieval": ("finetuned_retrieval", "4 · fine-tuned + retrieval"),
    "repair": ("repair", "5 · fine-tuned + repair"),
}


class ReplayOnly(TextToSQLModel):
    """Never generates; every prediction must come from the checkpoint."""

    model_id = DEFAULT_MODEL_ID

    def __init__(self, source: str) -> None:
        self.source = source
        self.misses = 0

    def generate(self, question: str, schema: str) -> GenerationResult:
        self.misses += 1
        return GenerationResult(sql="", raw_output="", latency_ms=0.0, ok=False,
                                error="no prediction on disk for this example")

    def describe(self) -> dict[str, Any]:
        return {"kind": "rescored_from_checkpoint", "model_id": DEFAULT_MODEL_ID,
                "source_generations": self.source}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_version_argument(p)
    p.add_argument("--config", choices=sorted(CONFIGS), action="append",
                   help="configuration(s) to re-score; repeatable")
    p.add_argument("--all", action="store_true", help="re-score every configuration")
    return p.parse_args()


def rescore_one(name: str, version, conn, schema_text, schema_fp, schema_info) -> dict[str, Any]:
    src_dir, label = CONFIGS[name]
    source = get_version("v1").experiments_root / src_dir / "generations.jsonl"
    if not source.exists():
        raise FileNotFoundError(source)

    out_dir = version.experiments_root / f"{name}_rescored"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "generations.jsonl"
    shutil.copyfile(source, checkpoint)

    test_set = version.split("test")
    examples = load_examples(test_set)
    v1_examples = {e["id"]: e for e in load_examples(get_version("v1").split("test"))}
    reworded = [e["id"] for e in examples if v1_examples[e["id"]]["question"] != e["question"]]

    model = ReplayOnly(str(source.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    metadata = build_run_metadata(model, test_set, conn, schema_fp, schema_text,
                                  schema_info.tables)
    metadata["rescoring"] = {
        "benchmark_version": version.name,
        "source_configuration": label,
        "source_generations": model.source,
        "generated_with_prompt": "v1",
        "generated_against_questions": "v1",
        "questions_reworded_in_this_version": reworded,
        "note": ("Predictions replayed from the v1 checkpoint and scored against "
                 f"benchmark {version.name} gold. Nothing was generated."),
    }

    print(f"\n--- {label}: {len(examples)} examples, "
          f"{len(reworded)} reworded questions scored on old-wording answers")

    def progress(done: int, total: int) -> None:
        if done % 100 == 0 or done == total:
            print(f"  {done:>4}/{total}", flush=True)

    results, drift = run_benchmark(
        model, examples, conn, schema_text, schema_info,
        workers=1, progress=progress, checkpoint_path=checkpoint, resume=True,
    )
    if model.misses:
        print(f"  [warning] {model.misses} examples had no checkpointed prediction")

    summary = summarise(results)
    summary["run_metadata"] = metadata
    if drift:
        summary["gold_fingerprint_drift"] = drift

    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    t, e, pa = summary["totals"], summary["error_rates"], summary["projection_analysis"]
    print(f"  strict {t['execution_accuracy_pct']:.2f} %   projection-tolerant "
          f"{pa['projection_tolerant_accuracy_pct']:.2f} %   executable "
          f"{t['executable_sql_pct']:.2f} %   hallucination "
          f"{e['schema_hallucination_pct']:.2f} %")
    return summary


def main() -> int:
    args = parse_args()
    version = get_version(args.version)
    if version.is_v1:
        print("[error] re-scoring v1 against v1 is the original run; pick --version v2",
              file=sys.stderr)
        return 2
    names = sorted(CONFIGS) if args.all else (args.config or [])
    if not names:
        print("[error] pass --all or --config <name>", file=sys.stderr)
        return 2
    if not version.split("test").exists():
        print(f"[error] {version.split('test')} missing; run generate_benchmark.py "
              f"and validate_dataset.py with --version {version.name}", file=sys.stderr)
        return 2

    try:
        cfg = app_config()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    print("=" * 72)
    print(f"RE-SCORE v1 PREDICTIONS AGAINST BENCHMARK {version.name.upper()}")
    print("=" * 72)

    v1_summaries = {}
    rows = []
    with read_only_connection(cfg) as conn:
        schema_text, schema_fp = build_schema_context(conn)
        schema_info = load_schema_info(conn)
        for name in names:
            summary = rescore_one(name, version, conn, schema_text, schema_fp, schema_info)
            v1_path = get_version("v1").experiments_root / CONFIGS[name][0] / "summary.json"
            v1 = json.loads(v1_path.read_text(encoding="utf-8")) if v1_path.exists() else None
            v1_summaries[name] = v1
            rows.append((CONFIGS[name][1],
                         v1["totals"]["execution_accuracy_pct"] if v1 else None,
                         summary["totals"]["execution_accuracy_pct"],
                         v1["projection_analysis"]["projection_tolerant_accuracy_pct"] if v1 else None,
                         summary["projection_analysis"]["projection_tolerant_accuracy_pct"]))

    print("\n" + "=" * 72)
    print(f"{'configuration':<30}{'v1 strict':>11}{'v2 strict':>11}{'v1 proj':>10}{'v2 proj':>10}")
    for label, a, b, c, d in rows:
        f = lambda v: f"{v:.2f}" if v is not None else "—"
        print(f"{label:<30}{f(a):>11}{f(b):>11}{f(c):>10}{f(d):>10}")
    print("=" * 72)
    print(f"written under {version.experiments_root.relative_to(PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
