"""Expand question/SQL templates into the benchmark dataset.

Phase 2.

Reads the templates in dataset/generation/templates.py, fills their slots with
real values sampled from the database, and writes one JSON object per line to
dataset/generated/benchmark.jsonl.

Nothing here checks that the SQL is correct — that is Phase 3's job, and it does
it by executing every query. Keeping generation and validation separate matters:
a generator that also validated its own output would be marking its own homework.

Deliberately not padded to a round number. Templates with no slots — "What is
the average order value?" — genuinely have only one instance, and are emitted
once per phrasing rather than duplicated to inflate the count. Volume without
information does not improve a benchmark.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/generate_benchmark.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from math import prod
from pathlib import Path
from typing import Any, Iterator

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.generation.templates import (  # noqa: E402
    ALL_TEMPLATES,
    Template,
    validate_all,
)
from dataset.generation.value_pools import (  # noqa: E402
    ValuePools,
    load_value_pools,
    missing_pools,
    sql_escape,
)
from src.sql.config import PROJECT_ROOT, ConfigError, app_config  # noqa: E402

RANDOM_SEED = 20260808
# Ceiling on how many value combinations any one template contributes. Without
# it, a template with three high-cardinality slots would dominate the dataset
# purely because its slots multiply out.
MAX_COMBINATIONS_PER_TEMPLATE = 120

OUTPUT_PATH = PROJECT_ROOT / "dataset" / "generated" / "benchmark.jsonl"
STATS_PATH = PROJECT_ROOT / "dataset" / "generated" / "statistics.json"


def slot_combinations(
    template: Template, pools: ValuePools, rng: random.Random
) -> list[dict[str, Any]]:
    """Return distinct slot-value assignments for one template.

    If the template has few possible combinations they are all used; otherwise a
    random sample is taken so no single template floods the dataset.
    """
    slots = template.slots
    if not slots:
        return [{}]

    sizes = [len(pools[s]) for s in slots]
    total = prod(sizes)

    if total <= MAX_COMBINATIONS_PER_TEMPLATE:
        # Small enough to enumerate exhaustively.
        combos: list[dict[str, Any]] = [{}]
        for slot in slots:
            combos = [{**c, slot: v} for c in combos for v in pools[slot]]
        rng.shuffle(combos)
        return combos

    seen: set[tuple] = set()
    combos = []
    # Bounded attempts: with a large combination space, collisions are rare, and
    # the cap stops the loop spinning if the space is smaller than it looks.
    for _ in range(MAX_COMBINATIONS_PER_TEMPLATE * 20):
        if len(combos) >= MAX_COMBINATIONS_PER_TEMPLATE:
            break
        choice = tuple(rng.choice(pools[s]) for s in slots)
        if choice in seen:
            continue
        seen.add(choice)
        combos.append(dict(zip(slots, choice)))
    return combos


def render(template: Template, values: dict[str, Any], phrasing: int) -> dict[str, Any]:
    """Produce one benchmark example from a template and a slot assignment."""
    question_template = template.questions[phrasing % len(template.questions)]

    # The question gets raw values; the SQL gets quote-escaped ones, because
    # only the SQL embeds them inside string literals.
    question = question_template.format(**values)
    sql = template.sql.format(**{k: sql_escape(v) for k, v in values.items()})

    return {
        "template_id": template.template_id,
        "question": question,
        "sql": sql,
        "difficulty": template.difficulty,
        "domain": template.domain,
        "query_type": list(template.query_type),
        "phrasing_index": phrasing % len(template.questions),
        "slot_values": {k: str(v) for k, v in values.items()},
    }


def generate(pools: ValuePools, rng: random.Random) -> Iterator[dict[str, Any]]:
    """Yield every benchmark example, in template order."""
    for template in ALL_TEMPLATES:
        for values in slot_combinations(template, pools, rng):
            # Every phrasing of every combination. The SQL repeats across
            # phrasings, which is the point: the model must learn that three
            # ways of asking the same thing map to one query.
            for phrasing in range(len(template.questions)):
                yield render(template, values, phrasing)


def build_statistics(examples: list[dict[str, Any]]) -> dict[str, Any]:
    difficulty = Counter(e["difficulty"] for e in examples)
    domain = Counter(e["domain"] for e in examples)
    features = Counter(f for e in examples for f in e["query_type"])
    per_template = Counter(e["template_id"] for e in examples)

    return {
        "total_examples": len(examples),
        "total_templates": len(ALL_TEMPLATES),
        "unique_questions": len({e["question"] for e in examples}),
        "unique_sql": len({e["sql"] for e in examples}),
        "by_difficulty": dict(difficulty.most_common()),
        "by_domain": dict(domain.most_common()),
        "by_query_feature": dict(features.most_common()),
        "examples_per_template": {
            "min": min(per_template.values()),
            "max": max(per_template.values()),
        },
        "random_seed": RANDOM_SEED,
        "max_combinations_per_template": MAX_COMBINATIONS_PER_TEMPLATE,
    }


def main() -> int:
    validate_all()

    try:
        cfg = app_config()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    try:
        with psycopg.connect(cfg.conninfo(), connect_timeout=10) as conn:
            pools = load_value_pools(conn)
    except psycopg.Error as exc:
        print(f"[error] could not read value pools: {exc}", file=sys.stderr)
        return 1

    # Catch a template referring to a slot with no value pool before generating
    # thousands of examples that would all be broken the same way.
    problems: list[str] = []
    for template in ALL_TEMPLATES:
        absent = missing_pools(template.slots, pools)
        if absent:
            problems.append(f"  {template.template_id}: unknown slots {absent}")
    if problems:
        print("[error] templates reference slots with no value pool:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1

    rng = random.Random(RANDOM_SEED)
    examples = list(generate(pools, rng))
    for i, example in enumerate(examples, start=1):
        example["id"] = f"bench-{i:06d}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    stats = build_statistics(examples)
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Templates          : {stats['total_templates']}")
    print(f"Examples generated : {stats['total_examples']:,}")
    print(f"Unique questions   : {stats['unique_questions']:,}")
    print(f"Unique SQL queries : {stats['unique_sql']:,}")
    print("\nBy difficulty:")
    for name, count in stats["by_difficulty"].items():
        print(f"  {name:<12}{count:>8,}")
    print("\nBy domain:")
    for name, count in stats["by_domain"].items():
        print(f"  {name:<12}{count:>8,}")
    print(f"\nWritten to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
