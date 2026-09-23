"""Build the static showcase Space from the frozen experiment results.

Phase 14.

Hugging Face now restricts free accounts to **static** Spaces — hosting Gradio
or Docker requires PRO — so live inference is not available. That constraint
turns out to suit this project: what makes the work worth showing is not that a
model emits SQL, it is the *evaluation*. A static explorer over all 453 scored
predictions shows more than a live demo would.

For every held-out question it carries: the question, the gold SQL, what each
measured configuration actually produced, and the outcome class the harness
assigned. Nothing is recomputed here — every field is read from the frozen
`experiments/` results, so the page cannot disagree with the benchmark.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/build_showcase.py
"""

from __future__ import annotations

import json
import shutil
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_versions import add_version_argument, get as get_version  # noqa: E402
from src.sql.config import PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "deploy" / "showcase"

# Configuration key -> (directory under the version's experiments root, label).
# A configuration with no results on this version is simply skipped, so the
# page can never show a row the benchmark did not measure.
CONFIGS = [
    ("base", "baseline", "Base model"),
    ("ft", "finetuned", "Fine-tuned"),
    ("ftr", "finetuned_retrieval", "Fine-tuned + retrieval"),
    ("repair", "repair", "Fine-tuned + repair"),
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the showcase data file")
    add_version_argument(parser)
    args = parser.parse_args()
    version = get_version(args.version)
    experiments = version.experiments_root

    gold = {r["id"]: r for r in load_jsonl(version.split("test"))}
    print(f"benchmark       : {version.name}")
    print(f"test set        : {len(gold)} questions")

    runs: dict[str, dict[str, dict]] = {}
    summaries: dict[str, dict] = {}
    for key, directory, label in CONFIGS:
        base = experiments / directory
        rows = load_jsonl(base / "results.jsonl")
        runs[key] = {r["example_id"]: r for r in rows}
        summary_path = base / "summary.json"
        if summary_path.exists():
            summaries[key] = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"  {label:<24}{len(rows):>4} results")

    examples = []
    for eid, g in gold.items():
        row = {
            "id": eid,
            "q": g["question"],
            "d": g["difficulty"],
            "gold": g["sql"],
            "tables": g.get("referenced_tables", []),
        }
        for key, _, _ in CONFIGS:
            r = runs[key].get(eid)
            if r is None:
                continue
            row[key] = {
                "sql": r.get("predicted_sql") or "",
                "ok": bool(r.get("correct")),
                "outcome": r.get("outcome"),
                "err": (r.get("error_message") or "")[:220],
                "rows": r.get("predicted_row_count"),
            }
        examples.append(row)

    # Deterministic order: hardest first, so the explorer opens on the
    # interesting cases rather than on 126 near-identical easy ones.
    tier_rank = {"enterprise": 0, "very_hard": 1, "hard": 2, "medium": 3, "easy": 4}
    examples.sort(key=lambda e: (tier_rank.get(e["d"], 9), e["id"]))

    def pct(key: str, path: tuple[str, ...]) -> float | None:
        node = summaries.get(key)
        for p in path:
            node = node.get(p) if isinstance(node, dict) else None
            if node is None:
                return None
        return node

    data = {
        "meta": {
            "n": len(examples),
            "gold_row_counts": {eid: g["execution"].get("row_count")
                                for eid, g in gold.items()
                                if isinstance(g.get("execution"), dict)},
        },
        "configs": [{"key": k, "label": lbl} for k, _, lbl in CONFIGS],
        "metrics": {
            k: {
                "accuracy": pct(k, ("totals", "execution_accuracy_pct")),
                "executable": pct(k, ("totals", "executable_sql_pct")),
                "hallucination": pct(k, ("error_rates", "schema_hallucination_pct")),
                "projection": pct(k, ("projection_analysis",
                                      "projection_tolerant_accuracy_pct")),
            } for k, _, _ in CONFIGS if k in summaries
        },
        "examples": examples,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (OUT / "data.json").write_text(blob, encoding="utf-8")

    print()
    print(f"  data.json  {len(blob) / 1024:,.0f} KB  "
          f"({len(examples)} examples x {len(summaries)} configurations)")

    by_tier: dict[str, int] = {}
    for e in examples:
        by_tier[e["d"]] = by_tier.get(e["d"], 0) + 1
    print("  tiers:", ", ".join(f"{k} {v}" for k, v in sorted(by_tier.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
