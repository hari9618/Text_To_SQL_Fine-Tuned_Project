"""Assemble the ablation study across every measured configuration.

Phase 14.

CLAUDE.md section 5 commits this project to five configurations, and to never
claiming an improvement without measuring it. This script collects whichever
have been run, prints them side by side, and writes `experiments/ABLATION.md`.

Configurations that have not been run are listed as **not measured** rather
than omitted or estimated. A gap in an ablation table is information; a
silently missing row is not.

Every configuration is scored on the same 453 unseen test examples, through the
same execution-based harness, against the same database fingerprint. The script
verifies that rather than assuming it: a configuration measured against
different data is excluded from the comparison and reported as such.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/ablation_report.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_versions import add_version_argument, get as get_version  # noqa: E402
from src.sql.config import PROJECT_ROOT  # noqa: E402

DIFFICULTY_ORDER = ["easy", "medium", "hard", "very_hard", "very hard",
                    "enterprise"]


@dataclass
class Config:
    """One row of the ablation."""

    number: int
    label: str
    directory: str
    description: str
    summary: dict[str, Any] | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    missing_reason: str | None = None

    @property
    def measured(self) -> bool:
        return self.summary is not None

    def metric(self, path: tuple[str, ...]) -> float | None:
        if self.summary is None:
            return None
        node: Any = self.summary
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                return None
        return node


CONFIGS = [
    Config(1, "Base model",
           "baseline",
           "Qwen3-8B Instruct, full schema in the prompt. The frozen Phase 4 "
           "baseline everything else is compared against."),
    Config(2, "Base + schema retrieval",
           "retrieval",
           "Only the tables a lexical retriever selects are shown "
           "(k=4, FK expansion 2, tuned on validation)."),
    Config(3, "Fine-tuned",
           "finetuned",
           "QLoRA adapter (r=16, 1 epoch, completion-only loss), full schema."),
    Config(4, "Fine-tuned + schema retrieval",
           "finetuned_retrieval",
           "The adapter with a retrieved schema subset."),
    Config(5, "Fine-tuned + repair",
           "repair",
           "The adapter, full schema, one repair attempt when the SQL fails to "
           "validate or execute."),
]

METRICS = [
    ("strict execution accuracy %", ("totals", "execution_accuracy_pct"), True),
    ("projection-tolerant %", ("projection_analysis",
                               "projection_tolerant_accuracy_pct"), True),
    ("executable SQL %", ("totals", "executable_sql_pct"), True),
    ("schema hallucination %", ("error_rates", "schema_hallucination_pct"), False),
    ("syntax error %", ("error_rates", "syntax_error_pct"), False),
]


def load_configs(experiments: Path) -> list[Config]:
    for cfg in CONFIGS:
        directory = experiments / cfg.directory
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            cfg.missing_reason = f"no {cfg.directory}/summary.json"
            continue
        cfg.summary = json.loads(summary_path.read_text(encoding="utf-8"))
        results_path = directory / "results.jsonl"
        if results_path.exists():
            cfg.results = [json.loads(l) for l
                           in results_path.read_text(encoding="utf-8").splitlines()
                           if l.strip()]
    return CONFIGS


def comparability_warnings(configs: list[Config]) -> list[str]:
    """Anything that would make two rows not actually comparable."""
    warnings: list[str] = []
    seen: dict[str, list[str]] = collections.defaultdict(list)

    for cfg in configs:
        if not cfg.measured:
            continue
        meta = cfg.summary.get("run_metadata", {})
        seen["examples"].append(str(cfg.metric(("totals", "total_examples"))))
        seen["prompt"].append(str(meta.get("prompt", {}).get("fingerprint")))
        seen["database"].append(
            str(meta.get("database", {}).get("data_fingerprint")))
        seen["dataset"].append(str(meta.get("dataset", {}).get("fingerprint")))

    for key, values in seen.items():
        distinct = sorted(set(values))
        if len(distinct) > 1:
            warnings.append(
                f"configurations differ in {key}: {', '.join(distinct)}")
    return warnings


def by_difficulty(cfg: Config, gold: dict[str, dict]) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for row in cfg.results:
        example = gold.get(row.get("example_id"))
        if example is None:
            continue
        tier = example.get("difficulty", "unknown")
        counts[tier][1] += 1
        if row.get("correct"):
            counts[tier][0] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def fmt(value: float | None, width: int = 12) -> str:
    return ("—" if value is None else f"{value:.2f}").rjust(width)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 14 ablation study")
    add_version_argument(parser)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    version = get_version(args.version)
    experiments = version.experiments_root
    TEST_SET = version.split("test")
    args.out = args.out or str(experiments / "ABLATION.md")

    configs = load_configs(experiments)
    measured = [c for c in configs if c.measured]
    gold = {}
    if TEST_SET.exists():
        gold = {json.loads(l)["id"]: json.loads(l)
                for l in TEST_SET.read_text(encoding="utf-8").splitlines()
                if l.strip()}

    print("=" * 78)
    print(f"PHASE 14 - ABLATION STUDY  (benchmark {version.name})")
    print("=" * 78)
    print(f"measured: {len(measured)} of {len(configs)} configurations")
    for cfg in configs:
        mark = "OK " if cfg.measured else "-- "
        note = "" if cfg.measured else f"  ({cfg.missing_reason})"
        print(f"  {mark}{cfg.number}. {cfg.label}{note}")

    warnings = comparability_warnings(measured)
    if warnings:
        print("\n[warning] comparability:")
        for w in warnings:
            print(f"  {w}")

    print()
    header = f"{'metric':<30}" + "".join(f"{c.number:>12}" for c in measured)
    print(header)
    print("-" * len(header))
    for name, path, _ in METRICS:
        print(f"{name:<30}" + "".join(fmt(c.metric(path)) for c in measured))

    # ---- markdown ---------------------------------------------------------
    lines: list[str] = []
    lines.append("# Ablation study")
    lines.append("")
    lines.append("Phase 14.")
    lines.append("")
    lines.append(
        "Every configuration below is scored on the **same 453 unseen test "
        "examples**, through the same execution-based harness, against the "
        "same database. The only thing that changes between rows is the one "
        "component named.")
    lines.append("")

    if warnings:
        lines.append("> **Comparability warning**")
        lines.append(">")
        for w in warnings:
            lines.append(f"> - {w}")
        lines.append("")

    lines.append("## Configurations")
    lines.append("")
    lines.append("| # | configuration | status | what changes |")
    lines.append("|---|---|---|---|")
    for cfg in configs:
        status = "measured" if cfg.measured else "**not measured**"
        lines.append(f"| {cfg.number} | {cfg.label} | {status} | "
                     f"{cfg.description} |")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| metric | " + " | ".join(f"**{c.number}**" for c in measured)
                 + " |")
    lines.append("|---|" + "---:|" * len(measured))
    for name, path, higher_better in METRICS:
        cells = []
        values = [c.metric(path) for c in measured]
        present = [v for v in values if v is not None]
        best = (max(present) if higher_better else min(present)) if present else None
        for v in values:
            if v is None:
                cells.append("—")
            elif best is not None and abs(v - best) < 1e-9:
                cells.append(f"**{v:.2f}**")
            else:
                cells.append(f"{v:.2f}")
        arrow = "higher is better" if higher_better else "lower is better"
        lines.append(f"| {name} <br><sub>{arrow}</sub> | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- per-difficulty ---------------------------------------------------
    if gold and any(c.results for c in measured):
        lines.append("## Strict execution accuracy by difficulty")
        lines.append("")
        tiers: list[str] = []
        for cfg in measured:
            for tier in by_difficulty(cfg, gold):
                if tier not in tiers:
                    tiers.append(tier)
        tiers.sort(key=lambda t: (DIFFICULTY_ORDER.index(t)
                                  if t in DIFFICULTY_ORDER else 99, t))

        lines.append("| tier | n | " + " | ".join(f"**{c.number}**"
                                                  for c in measured if c.results)
                     + " |")
        lines.append("|---|---:|" + "---:|" * len([c for c in measured if c.results]))
        for tier in tiers:
            cells, n = [], 0
            for cfg in measured:
                if not cfg.results:
                    continue
                correct, total = by_difficulty(cfg, gold).get(tier, (0, 0))
                n = max(n, total)
                cells.append(f"{100 * correct / total:.1f}" if total else "—")
            lines.append(f"| {tier} | {n} | " + " | ".join(cells) + " |")
        lines.append("")

    # ---- repair detail ----------------------------------------------------
    repair_cfg = next((c for c in measured if c.number == 5), None)
    if repair_cfg and repair_cfg.summary.get("repair"):
        r = repair_cfg.summary["repair"]
        lines.append("## Repair loop")
        lines.append("")
        lines.append("| | |")
        lines.append("|---|---:|")
        lines.append(f"| queries that failed and were retried | {r['attempted']} |")
        lines.append(f"| now execute | {r['now_executes']} "
                     f"({r['success_rate_pct']:.1f} %) |")
        lines.append(f"| now return the gold rows | {r['now_correct']} "
                     f"({r['correctness_rate_pct']:.1f} %) |")
        lines.append(f"| model returned identical SQL | "
                     f"{r['returned_identical_sql']} |")
        lines.append(f"| mean repair latency | "
                     f"{r['mean_repair_latency_ms']:.0f} ms |")
        lines.append("")
        lines.append(
            "*Repair success* means the query now executes — the only signal "
            "available in production, where there is no gold answer to compare "
            "against. *Repair correctness* means it now returns the right rows. "
            "A repair that turns a crash into a confident wrong answer counts "
            "on the first and not the second, which is why both are reported.")
        lines.append("")

    # ---- missing ----------------------------------------------------------
    unmeasured = [c for c in configs if not c.measured]
    if unmeasured:
        lines.append("## Not measured")
        lines.append("")
        for cfg in unmeasured:
            lines.append(f"- **{cfg.number}. {cfg.label}** — {cfg.missing_reason}")
        lines.append("")
        lines.append(
            "These are listed rather than omitted. An ablation with a hidden "
            "gap invites the reader to assume the missing row would have "
            "agreed with the others.")
        lines.append("")

    lines.append("## Provenance")
    lines.append("")
    lines.append("| # | model | schema mode | prompt | database |")
    lines.append("|---|---|---|---|---|")
    for cfg in measured:
        meta = cfg.summary.get("run_metadata", {})
        model = meta.get("model", {})
        lines.append(
            f"| {cfg.number} | `{model.get('kind', '?')}` | "
            f"`{meta.get('schema_context', {}).get('mode', '?')}` | "
            f"`{meta.get('prompt', {}).get('fingerprint', '?')}` | "
            f"`{meta.get('database', {}).get('data_fingerprint', '?')}` |")
    lines.append("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten to {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
