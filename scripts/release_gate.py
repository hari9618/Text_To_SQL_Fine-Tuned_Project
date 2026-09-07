"""Decide whether a fine-tuned model is fit to deploy.

Phase 14.

A benchmark reports numbers. A gate *refuses a release*. This is the second
thing, and the difference matters: Phase 10 measured that fine-tuning improved
strict execution accuracy by 40 points **and** that it made executable SQL and
schema hallucination worse. Reporting both is honest; shipping anyway without
anyone having to acknowledge the regression is not.

The gate is deliberately hard to pass by accident:

* **Improvement is not enough.** A candidate that raises accuracy while
  breaking more queries fails, unless the regression is inside an explicit
  tolerance.
* **Thresholds live here, in version control**, not in someone's head. Changing
  one is a reviewable diff.
* **A missing measurement is a failure, not a pass.** An unmeasured
  configuration cannot be released on the assumption that it is fine.

Exit code 0 means releasable; 1 means blocked. That makes it usable as a CI
step.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/release_gate.py
    env\\Scripts\\python.exe scripts/release_gate.py --candidate experiments/repair
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sql.config import PROJECT_ROOT  # noqa: E402

BASELINE = PROJECT_ROOT / "experiments" / "baseline" / "summary.json"
DEFAULT_CANDIDATE = PROJECT_ROOT / "experiments" / "finetuned"

# The frozen contract. Every value here is a decision someone has to defend.
MIN_TEST_EXAMPLES = 453          # the full held-out split, not a sample
MIN_ACCURACY_GAIN_PP = 5.0       # below this the fine-tune is not worth shipping
MAX_EXECUTABLE_DROP_PP = 4.0     # more broken queries than this blocks release
MAX_HALLUCINATION_RISE_PP = 2.0  # inventing tables/columns is the worst failure
MAX_SYNTAX_ERROR_PCT = 1.0
EXPECTED_PROMPT_FP = "8288e41a496531a9"
EXPECTED_SCHEMA_FP = "d03619e711661bc5"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    blocking: bool = True


def metric(summary: dict[str, Any], path: tuple[str, ...]) -> float | None:
    node: Any = summary
    for key in path:
        node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            return None
    return node


ACCURACY = ("totals", "execution_accuracy_pct")
EXECUTABLE = ("totals", "executable_sql_pct")
HALLUCINATION = ("error_rates", "schema_hallucination_pct")
SYNTAX = ("error_rates", "syntax_error_pct")


MAX_TIER_REGRESSION_PP = 5.0   # a tier may not collapse while the mean improves


def accuracy_by_tier(results_path: Path, gold: dict[str, dict]) -> dict[str, float]:
    """Strict accuracy per difficulty tier, or {} if results are unavailable."""
    if not results_path.exists():
        return {}
    hits: dict[str, list[int]] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        example = gold.get(row.get("example_id"))
        if example is None:
            continue
        tier = example.get("difficulty", "unknown")
        bucket = hits.setdefault(tier, [0, 0])
        bucket[1] += 1
        if row.get("correct"):
            bucket[0] += 1
    return {t: 100 * c / n for t, (c, n) in hits.items() if n}


def run_gate(base: dict, cand: dict, tests_passed: bool | None,
             tier_base: dict[str, float] | None = None,
             tier_cand: dict[str, float] | None = None) -> list[Check]:
    checks: list[Check] = []

    # ---- the comparison must be legitimate before the numbers mean anything
    n = metric(cand, ("totals", "total_examples")) or 0
    checks.append(Check(
        "evaluated on the full held-out split",
        n >= MIN_TEST_EXAMPLES,
        f"{n:.0f} examples (need {MIN_TEST_EXAMPLES})"))

    meta = cand.get("run_metadata", {})
    prompt_fp = meta.get("prompt", {}).get("fingerprint")
    checks.append(Check(
        "prompt matches the frozen baseline",
        prompt_fp == EXPECTED_PROMPT_FP,
        f"{prompt_fp} (need {EXPECTED_PROMPT_FP})"))

    db_cand = meta.get("database", {}).get("data_fingerprint")
    db_base = base.get("run_metadata", {}).get("database", {}).get("data_fingerprint")
    checks.append(Check(
        "scored against the same database as the baseline",
        db_cand is not None and db_cand == db_base,
        f"{db_cand} vs baseline {db_base}"))

    # ---- did it actually get better
    gain = (metric(cand, ACCURACY) or 0) - (metric(base, ACCURACY) or 0)
    checks.append(Check(
        "accuracy gain clears the bar",
        gain >= MIN_ACCURACY_GAIN_PP,
        f"{gain:+.2f} pp (need >= {MIN_ACCURACY_GAIN_PP})"))

    # ---- did it break anything on the way
    drop = (metric(base, EXECUTABLE) or 0) - (metric(cand, EXECUTABLE) or 0)
    checks.append(Check(
        "executable SQL did not regress too far",
        drop <= MAX_EXECUTABLE_DROP_PP,
        f"-{drop:.2f} pp (tolerance {MAX_EXECUTABLE_DROP_PP})"))

    rise = (metric(cand, HALLUCINATION) or 0) - (metric(base, HALLUCINATION) or 0)
    checks.append(Check(
        "schema hallucination did not rise too far",
        rise <= MAX_HALLUCINATION_RISE_PP,
        f"{rise:+.2f} pp (tolerance {MAX_HALLUCINATION_RISE_PP})"))

    syntax = metric(cand, SYNTAX) or 0
    checks.append(Check(
        "syntax errors within budget",
        syntax <= MAX_SYNTAX_ERROR_PCT,
        f"{syntax:.2f} % (max {MAX_SYNTAX_ERROR_PCT})"))

    # ---- the harness itself has to be trustworthy
    if tests_passed is not None:
        checks.append(Check(
            "test suite passes (includes adversarial security tests)",
            tests_passed, "pytest tests/" ))

    # ---- a mean can improve while one tier collapses
    if tier_base and tier_cand:
        regressed = [
            f"{tier} {tier_cand[tier] - tier_base[tier]:+.1f}pp"
            for tier in sorted(set(tier_base) & set(tier_cand))
            if tier_base[tier] - tier_cand[tier] > MAX_TIER_REGRESSION_PP
        ]
        checks.append(Check(
            "no difficulty tier collapsed",
            not regressed,
            ", ".join(regressed) if regressed
            else f"{len(set(tier_base) & set(tier_cand))} tiers compared, "
                 f"tolerance {MAX_TIER_REGRESSION_PP} pp"))
    else:
        checks.append(Check(
            "per-tier comparison available",
            False,
            "results.jsonl missing on one side - cannot rule out a tier collapse",
            blocking=False))

    return checks


def main() -> int:
    p = argparse.ArgumentParser(description="Release gate for a fine-tuned model")
    p.add_argument("--candidate", default=str(DEFAULT_CANDIDATE),
                   help="experiment directory holding summary.json")
    p.add_argument("--baseline", default=str(BASELINE))
    p.add_argument("--skip-tests", action="store_true",
                   help="do not run pytest as part of the gate")
    args = p.parse_args()

    def resolve(raw: str) -> Path:
        """Accept a path relative to the project root or the shell's cwd."""
        path = Path(raw)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path) if not path.exists() else path.resolve()
        if path.is_dir():
            path = path / "summary.json"
        return path

    cand_path = resolve(args.candidate)
    base_path = resolve(args.baseline)

    def show(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    print("=" * 74)
    print("RELEASE GATE")
    print("=" * 74)

    for path, label in ((base_path, "baseline"), (cand_path, "candidate")):
        if not path.exists():
            print(f"[BLOCKED] {label} summary not found: {path}\n"
                  f"          An unmeasured configuration is not releasable.",
                  file=sys.stderr)
            return 1

    base = json.loads(base_path.read_text(encoding="utf-8"))
    cand = json.loads(cand_path.read_text(encoding="utf-8"))

    print(f"baseline  : {show(base_path)}")
    print(f"candidate : {show(cand_path)}")
    print()

    tests_passed: bool | None = None
    if not args.skip_tests:
        print("running the test suite...")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q"],
            cwd=PROJECT_ROOT, capture_output=True, text=True)
        tests_passed = proc.returncode == 0
        tail = [l for l in proc.stdout.strip().splitlines() if l.strip()][-1:]
        print(f"  {tail[0] if tail else 'no output'}")
        print()

    gold: dict[str, dict] = {}
    test_set = PROJECT_ROOT / "dataset" / "test" / "test.jsonl"
    if test_set.exists():
        gold = {json.loads(l)["id"]: json.loads(l)
                for l in test_set.read_text(encoding="utf-8").splitlines()
                if l.strip()}
    tier_base = accuracy_by_tier(base_path.parent / "results.jsonl", gold)
    tier_cand = accuracy_by_tier(cand_path.parent / "results.jsonl", gold)

    checks = run_gate(base, cand, tests_passed, tier_base, tier_cand)

    width = max(len(c.name) for c in checks)
    for c in checks:
        mark = "PASS" if c.passed else ("FAIL" if c.blocking else "warn")
        print(f"  [{mark}] {c.name:<{width}}  {c.detail}")

    blocking_failures = [c for c in checks if c.blocking and not c.passed]

    print()
    print("-" * 74)
    print(f"{'metric':<28}{'baseline':>12}{'candidate':>12}{'change':>12}")
    print("-" * 74)
    for label, path, better_is_up in (
        ("strict execution accuracy", ACCURACY, True),
        ("executable SQL", EXECUTABLE, True),
        ("schema hallucination", HALLUCINATION, False),
        ("syntax errors", SYNTAX, False),
    ):
        b, c = metric(base, path), metric(cand, path)
        if b is None or c is None:
            continue
        delta = c - b
        direction = "better" if (delta > 0) == better_is_up else "worse"
        if abs(delta) < 0.005:
            direction = "same"
        print(f"{label:<28}{b:>11.2f}%{c:>11.2f}%{delta:>+11.2f}  {direction}")
    print("-" * 74)

    if tier_base and tier_cand:
        print()
        print(f"{'difficulty':<28}{'baseline':>12}{'candidate':>12}{'change':>12}")
        print("-" * 74)
        for tier in sorted(set(tier_base) & set(tier_cand),
                           key=lambda t: tier_cand[t] - tier_base[t]):
            d = tier_cand[tier] - tier_base[tier]
            print(f"{tier:<28}{tier_base[tier]:>11.1f}%{tier_cand[tier]:>11.1f}%"
                  f"{d:>+11.1f}")
        print("-" * 74)

    if blocking_failures:
        print(f"\nRELEASE BLOCKED - {len(blocking_failures)} failing check(s):")
        for c in blocking_failures:
            print(f"  - {c.name}: {c.detail}")
        print("\nRaising a threshold to pass is a reviewable change to this "
              "file, not a decision to make in the moment.")
        return 1

    print("\nRELEASE APPROVED")
    print("\nWhat this does NOT establish, and should be said out loud:")
    print("  - every test question came from the same generator as training;")
    print("    real user phrasing is untested")
    print("  - one seed, one run: no variance estimate")
    print("  - no load, latency or cost ceiling has been measured")
    print("  - no shadow or canary deployment has been run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
