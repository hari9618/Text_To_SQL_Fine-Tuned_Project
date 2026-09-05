"""Validate the generated benchmark and build leakage-free splits.

Phase 3.

Every generated example is put through the same gauntlet:

    question + SQL
          |
          v
    static analysis      parseable? one statement? read-only? real tables?
          |
          v
    slot representation  does the question actually mention the value the
          |              SQL filters on?
          v
    PostgreSQL execution does it run, and what does it return?
          |
          v
    duplicate/collision  same question with conflicting gold SQL or results?
          |
          v
    validated example

Survivors are split by *template equivalence group*, never by row. Splitting
rows at random would put three phrasings of the same query on both sides of the
train/test line, and the model would score well by memorising rather than
generalising — the reported accuracy would be inflated and worthless.

Nothing here is generated, padded, or relaxed. Rejections are recorded with a
reason and written out, so the cost of every rule is visible.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/validate_dataset.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.generation.templates import validate_all  # noqa: E402
from src.constants import DATA_AS_OF, DATA_AS_OF_SQL  # noqa: E402
from src.sql import validator  # noqa: E402
from src.sql.config import PROJECT_ROOT, ConfigError, app_config  # noqa: E402
from src.sql.executor import (  # noqa: E402
    ExecutionResult,
    execute,
    load_schema_info,
    read_only_connection,
)

RANDOM_SEED = 20260808
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
STATEMENT_TIMEOUT_MS = 30_000

INPUT_PATH = PROJECT_ROOT / "dataset" / "generated" / "benchmark.jsonl"
VALIDATED_DIR = PROJECT_ROOT / "dataset" / "validated"
SPLIT_DIRS = {
    "train": PROJECT_ROOT / "dataset" / "train",
    "validation": PROJECT_ROOT / "dataset" / "validation",
    "test": PROJECT_ROOT / "dataset" / "test",
}

# Human-readable explanation for every rejection category, carried into the
# report so the numbers are never bare.
REJECTION_REASONS = {
    "unparseable_sql": "SQL could not be parsed by sqlglot",
    "multiple_statements": "more than one SQL statement in a single example",
    "dangerous_sql": "statement is not read-only (INSERT/UPDATE/DELETE/DDL)",
    "unknown_table": "references a table that does not exist in the schema",
    "slot_not_in_question": (
        "SQL filters on a value the question never mentions, making the "
        "question unanswerable"
    ),
    "execution_syntax_error": "PostgreSQL rejected the SQL as malformed",
    "execution_undefined_table": "PostgreSQL: table does not exist",
    "execution_undefined_column": "PostgreSQL: column does not exist",
    "execution_undefined_function": "PostgreSQL: function does not exist",
    "execution_ambiguous_column": "PostgreSQL: column reference is ambiguous",
    "execution_timeout": f"query exceeded the {STATEMENT_TIMEOUT_MS} ms timeout",
    "execution_other": "PostgreSQL rejected the query for another reason",
    "empty_result": (
        "gold query returns zero rows, so any query returning nothing would "
        "score as correct"
    ),
    "duplicate_pair": "identical question and SQL already present",
    "conflicting_question": (
        "same question text maps to different gold SQL or different results"
    ),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_examples(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalise_sql(sql: str) -> str:
    """Whitespace- and case-insensitive key for identifying identical queries."""
    return " ".join(sql.split()).lower()


# ---------------------------------------------------------------------------
# Template equivalence
# ---------------------------------------------------------------------------

class UnionFind:
    """Minimal disjoint-set, used to group templates that must not be split."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def build_equivalence_groups(examples: list[dict[str, Any]]) -> dict[str, str]:
    """Map template_id -> group id, merging templates that share SQL.

    Two templates that emit byte-identical SQL are the same query wearing two
    ids. Splitting them apart would place the identical query in both train and
    test — leakage that template-level splitting alone would not catch.
    """
    uf = UnionFind()
    by_sql: dict[str, set[str]] = defaultdict(set)

    for example in examples:
        template_id = example["template_id"]
        uf.find(template_id)  # ensure every template exists as its own group
        by_sql[normalise_sql(example["sql"])].add(template_id)

    for template_ids in by_sql.values():
        template_ids = sorted(template_ids)
        for other in template_ids[1:]:
            uf.union(template_ids[0], other)

    return {t: uf.find(t) for t in uf.parent}


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

@dataclass
class Group:
    group_id: str
    template_ids: list[str]
    difficulty: str
    size: int


def assign_splits(groups: list[Group], rng: random.Random) -> dict[str, str]:
    """Assign whole groups to splits, stratified by difficulty.

    Groups are atomic: every example sharing a group lands in one split. Within
    each difficulty tier, groups are placed greedily into whichever split is
    furthest below its target share, largest group first, so the ratios come out
    close despite the group sizes being uneven.
    """
    assignment: dict[str, str] = {}

    by_difficulty: dict[str, list[Group]] = defaultdict(list)
    for group in groups:
        by_difficulty[group.difficulty].append(group)

    for difficulty, tier_groups in sorted(by_difficulty.items()):
        total = sum(g.size for g in tier_groups)
        targets = {s: total * r for s, r in SPLIT_RATIOS.items()}
        current = {s: 0 for s in SPLIT_RATIOS}

        rng.shuffle(tier_groups)
        # Largest first: placing big groups while every split still has room
        # keeps the final ratios closer to target.
        tier_groups.sort(key=lambda g: g.size, reverse=True)

        for group in tier_groups:
            split = max(current, key=lambda s: targets[s] - current[s])
            assignment[group.group_id] = split
            current[split] += group.size

    return assignment


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("PHASE 3 — DATASET VALIDATION")
    print("=" * 70)

    # --- 0. template-level validators (requirement 6) -----------------------
    print("\n[1/7] Template validators...")
    try:
        validate_all()
        print("      unique ids, slot coverage, question collisions: PASS")
        template_validators_passed = True
    except ValueError as exc:
        print(f"      FAIL: {exc}", file=sys.stderr)
        template_validators_passed = False
        return 1

    if not INPUT_PATH.exists():
        print(f"[error] {INPUT_PATH} not found. Run generate_benchmark.py first.",
              file=sys.stderr)
        return 2

    examples = load_examples(INPUT_PATH)
    print(f"      loaded {len(examples):,} generated examples")

    try:
        cfg = app_config()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    rejections: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []

    def reject(example: dict[str, Any], reason: str, detail: str = "") -> None:
        rejections.append({
            "id": example.get("id"),
            "template_id": example.get("template_id"),
            "difficulty": example.get("difficulty"),
            "question": example.get("question"),
            "sql": example.get("sql"),
            "reason": reason,
            "detail": detail,
        })

    with read_only_connection(cfg, STATEMENT_TIMEOUT_MS) as conn:
        schema = load_schema_info(conn)
        print(f"      schema: {len(schema.tables)} tables, "
              f"{sum(len(c) for c in schema.columns_by_table.values())} columns")

        # --- 1. static analysis (requirements 3, 4) -------------------------
        print("\n[2/7] Static analysis (parse, read-only, table grounding)...")
        unique_sql = sorted({e["sql"] for e in examples})
        static_by_sql = {
            sql: validator.analyse(sql, schema.tables) for sql in unique_sql
        }
        static_failures = sum(1 for s in static_by_sql.values() if not s.ok)
        print(f"      {len(unique_sql):,} unique queries analysed, "
              f"{static_failures} failed")

        # --- 2. execution (requirements 1, 2) -------------------------------
        print(f"\n[3/7] Executing {len(unique_sql):,} unique queries "
              f"(timeout {STATEMENT_TIMEOUT_MS} ms)...")
        exec_by_sql: dict[str, ExecutionResult] = {}
        for i, sql in enumerate(unique_sql, start=1):
            # Only execute what passed static analysis. Running a statement
            # already identified as a write would defeat the purpose of the
            # check, and the read-only session would reject it anyway.
            static = static_by_sql[sql]
            if static.parseable and static.is_read_only and static.statement_count == 1:
                exec_by_sql[sql] = execute(conn, sql)
            else:
                exec_by_sql[sql] = ExecutionResult(
                    ok=False, latency_ms=0.0, error_class="not_executed",
                    error_message="skipped: failed static analysis",
                )
            if i % 100 == 0 or i == len(unique_sql):
                ok = sum(1 for r in exec_by_sql.values() if r.ok)
                print(f"      {i:>5,}/{len(unique_sql):,}   ok={ok:,}")

    exec_failures = sum(1 for r in exec_by_sql.values() if not r.ok)
    print(f"      execution failures: {exec_failures} of {len(unique_sql):,} queries")

    # --- 3. per-example screening ------------------------------------------
    print("\n[4/7] Screening examples...")
    seen_pairs: set[tuple[str, str]] = set()

    for example in examples:
        sql = example["sql"]
        static = static_by_sql[sql]

        reason = static.failure_reason()
        if reason:
            reject(example, reason,
                   static.parse_error or ",".join(static.unknown_tables))
            continue

        slot_check = validator.check_slot_representation(
            example["question"], sql, example.get("slot_values", {})
        )
        if not slot_check.ok:
            reject(example, "slot_not_in_question", ",".join(slot_check.missing))
            continue

        result = exec_by_sql[sql]
        if not result.ok:
            reject(example, f"execution_{result.error_class}",
                   result.error_message or "")
            continue

        # A gold query returning nothing cannot separate a correct model answer
        # from a wrong one — both return nothing and both would score correct.
        if result.row_count == 0:
            reject(example, "empty_result", "gold query returned 0 rows")
            continue

        pair = (example["question"], normalise_sql(sql))
        if pair in seen_pairs:
            reject(example, "duplicate_pair", "identical question and SQL")
            continue
        seen_pairs.add(pair)

        example = dict(example)
        example["execution"] = result.as_dict()
        example["referenced_tables"] = list(static.referenced_tables)
        accepted.append(example)

    print(f"      {len(accepted):,} passed, {len(rejections):,} rejected so far")

    # --- 4. semantic collisions (requirements 7, 8) -------------------------
    print("\n[5/7] Duplicate and collision analysis...")
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in accepted:
        by_question[example["question"]].append(example)

    conflicting_questions: list[dict[str, Any]] = []
    conflict_ids: set[str] = set()
    for question, group in by_question.items():
        sqls = {normalise_sql(e["sql"]) for e in group}
        fingerprints = {e["execution"]["fingerprint"] for e in group}
        if len(sqls) > 1 or len(fingerprints) > 1:
            conflicting_questions.append({
                "question": question,
                "template_ids": sorted({e["template_id"] for e in group}),
                "distinct_sql": len(sqls),
                "distinct_results": len(fingerprints),
            })
            conflict_ids.update(e["id"] for e in group)

    if conflict_ids:
        survivors = []
        for example in accepted:
            if example["id"] in conflict_ids:
                reject(example, "conflicting_question",
                       "same question, different gold SQL or results")
            else:
                survivors.append(example)
        accepted = survivors

    duplicate_question_texts = sum(
        1 for group in by_question.values() if len(group) > 1
    )
    print(f"      questions appearing more than once: {duplicate_question_texts}")
    print(f"      conflicting questions: {len(conflicting_questions)}")

    # Different SQL producing identical results is reported, not rejected:
    # two genuinely distinct questions can legitimately share an answer.
    fingerprint_to_sql: dict[str, set[str]] = defaultdict(set)
    for example in accepted:
        fingerprint_to_sql[example["execution"]["fingerprint"]].add(
            normalise_sql(example["sql"])
        )
    result_collisions = {
        fp: sorted(sqls) for fp, sqls in fingerprint_to_sql.items() if len(sqls) > 1
    }
    print(f"      distinct queries sharing a result set: {len(result_collisions)}")

    # --- 5. splits (requirements 9, 10, 11) --------------------------------
    print("\n[6/7] Splitting by template equivalence group...")
    group_of_template = build_equivalence_groups(accepted)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in accepted:
        grouped[group_of_template[example["template_id"]]].append(example)

    groups = [
        Group(
            group_id=group_id,
            template_ids=sorted({e["template_id"] for e in rows}),
            difficulty=Counter(e["difficulty"] for e in rows).most_common(1)[0][0],
            size=len(rows),
        )
        for group_id, rows in grouped.items()
    ]
    merged_groups = [g for g in groups if len(g.template_ids) > 1]
    print(f"      {len(groups)} groups from "
          f"{len({e['template_id'] for e in accepted})} templates "
          f"({len(merged_groups)} merged by identical SQL)")

    rng = random.Random(RANDOM_SEED)
    split_of_group = assign_splits(groups, rng)

    splits: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLIT_RATIOS}
    for example in accepted:
        split = split_of_group[group_of_template[example["template_id"]]]
        example["split"] = split
        splits[split].append(example)

    for name, rows in splits.items():
        share = 100.0 * len(rows) / max(len(accepted), 1)
        print(f"      {name:<11}{len(rows):>6,}  ({share:.1f}%)")

    # --- 6. leakage verification (requirement 9) ---------------------------
    print("\n[7/7] Leakage checks...")

    def cross_split(key: str) -> list[str]:
        """Values that appear in more than one split."""
        where: dict[str, set[str]] = defaultdict(set)
        for split_name, rows in splits.items():
            for row in rows:
                value = row["question"] if key == "question" else (
                    normalise_sql(row["sql"]) if key == "sql" else row[key]
                )
                where[value].add(split_name)
        return sorted(v for v, s in where.items() if len(s) > 1)

    leaked_templates = cross_split("template_id")
    leaked_questions = cross_split("question")
    leaked_sql = cross_split("sql")
    leaked_groups = sorted(
        v for v, s in (
            (g, {split_of_group[g]}) for g in grouped
        ) if len(s) > 1
    )

    leakage = {
        "templates_in_multiple_splits": len(leaked_templates),
        "questions_in_multiple_splits": len(leaked_questions),
        "sql_in_multiple_splits": len(leaked_sql),
        "groups_in_multiple_splits": len(leaked_groups),
    }
    for name, count in leakage.items():
        status = "PASS" if count == 0 else "FAIL"
        print(f"      {name:<34}{count:>5}  {status}")

    leakage_detected = any(v > 0 for v in leakage.values())

    # --- 7. write outputs --------------------------------------------------
    VALIDATED_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(VALIDATED_DIR / "validated.jsonl", accepted)
    write_jsonl(VALIDATED_DIR / "rejected.jsonl", rejections)
    for name, directory in SPLIT_DIRS.items():
        write_jsonl(directory / f"{name}.jsonl", splits[name])

    # --- 8. report (requirement 12) ----------------------------------------
    per_template: dict[str, dict[str, Any]] = {}
    generated_counts = Counter(e["template_id"] for e in examples)
    valid_counts = Counter(e["template_id"] for e in accepted)
    rejected_by_template: dict[str, Counter] = defaultdict(Counter)
    for row in rejections:
        rejected_by_template[row["template_id"]][row["reason"]] += 1

    split_of_template = {
        t: split_of_group[g] for t, g in group_of_template.items()
    }
    for template_id, generated in sorted(generated_counts.items()):
        per_template[template_id] = {
            "generated": generated,
            "valid": valid_counts.get(template_id, 0),
            "rejected": generated - valid_counts.get(template_id, 0),
            "rejection_reasons": dict(rejected_by_template.get(template_id, {})),
            "split": split_of_template.get(template_id),
        }

    reason_counts = Counter(r["reason"] for r in rejections)
    # Recorded so a future evaluation run can confirm it is comparing against
    # gold results produced under the same time anchor.
    time_anchored = sum(
        1 for e in accepted if DATA_AS_OF_SQL.split("'")[1] in e["sql"]
    )
    still_using_now = sum(1 for e in accepted if "NOW()" in e["sql"].upper())

    report: dict[str, Any] = {
        "phase": 3,
        "random_seed": RANDOM_SEED,
        "data_as_of": DATA_AS_OF.isoformat(),
        "data_as_of_sql": DATA_AS_OF_SQL,
        "time_anchored_examples": time_anchored,
        "examples_still_using_now": still_using_now,
        "statement_timeout_ms": STATEMENT_TIMEOUT_MS,
        "split_ratios": SPLIT_RATIOS,
        "template_validators_passed": template_validators_passed,
        "totals": {
            "total_examples": len(examples),
            "valid_examples": len(accepted),
            "invalid_examples": len(rejections),
            "validation_rate_pct": round(100.0 * len(accepted) / len(examples), 2),
            "unique_sql_executed": len(unique_sql),
            "execution_failures_unique_sql": exec_failures,
            "duplicate_question_texts": duplicate_question_texts,
            "conflicting_questions": len(conflicting_questions),
            "result_set_collisions": len(result_collisions),
            "leakage_detected": leakage_detected,
        },
        "rejections_by_reason": {
            reason: {
                "count": count,
                "explanation": REJECTION_REASONS.get(reason, "unclassified"),
            }
            for reason, count in reason_counts.most_common()
        },
        "leakage_checks": leakage,
        "valid_by_difficulty": dict(
            Counter(e["difficulty"] for e in accepted).most_common()
        ),
        "valid_by_domain": dict(Counter(e["domain"] for e in accepted).most_common()),
        "splits": {
            name: {
                "examples": len(rows),
                "templates": len({r["template_id"] for r in rows}),
                "by_difficulty": dict(
                    Counter(r["difficulty"] for r in rows).most_common()
                ),
            }
            for name, rows in splits.items()
        },
        "merged_template_groups": [
            {"group": g.group_id, "templates": g.template_ids}
            for g in merged_groups
        ],
        "conflicting_question_details": conflicting_questions[:50],
        "by_template": per_template,
    }

    (VALIDATED_DIR / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"\nTime anchor: {DATA_AS_OF_SQL}")
    print(f"      examples anchored to it     : {time_anchored:,}")
    print(f"      examples still using NOW()  : {still_using_now}"
          f"{'  <-- DRIFT' if still_using_now else '  PASS'}")

    print("\n" + "=" * 70)
    print(f"VALID   {len(accepted):>6,} / {len(examples):,} "
          f"({report['totals']['validation_rate_pct']}%)")
    print(f"REJECT  {len(rejections):>6,}")
    for reason, count in reason_counts.most_common():
        print(f"        {reason:<28}{count:>6,}  "
              f"{REJECTION_REASONS.get(reason, '')[:40]}")
    print("=" * 70)
    print(f"\nWritten to {VALIDATED_DIR.relative_to(PROJECT_ROOT)}/ and split dirs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
