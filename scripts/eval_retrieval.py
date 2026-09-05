"""Measure schema retrieval on its own, without calling any model.

Phase 6, step 1.

Retrieval is evaluated before it is wired to the LLM, because the two failure
modes are completely different and mixing them makes both unreadable:

* **Retrieval recall** — did the selected subset contain every table the gold
  query needs? A miss here makes a correct answer *impossible*; the model can
  only respond by inventing a relationship. Recall is therefore the metric that
  matters, and it is far more important than precision.
* **Generation** — given a correct subset, did the model write correct SQL?

Ground truth comes from ``referenced_tables``, recorded per example during
Phase 3 by parsing each gold query. No new labelling was needed.

Costs nothing to run: no API calls, no tokens.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/eval_retrieval.py
    env\\Scripts\\python.exe scripts/eval_retrieval.py --max-tables 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.schema_context import build_schema_context  # noqa: E402
from src.retrieval.keyword import KeywordRetriever  # noqa: E402
from src.retrieval.schema_index import build_index  # noqa: E402
from src.sql.config import PROJECT_ROOT, ConfigError, app_config  # noqa: E402
from src.sql.executor import read_only_connection  # noqa: E402

TEST_SET = PROJECT_ROOT / "dataset" / "test" / "test.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "retrieval"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate schema retrieval offline")
    parser.add_argument("--max-tables", type=int, default=6)
    parser.add_argument("--expand", type=int, default=0,
                        help="also include direct FK neighbours of the top N seeds")
    parser.add_argument("--split", default=str(TEST_SET))
    parser.add_argument("--show-misses", type=int, default=8)
    args = parser.parse_args()

    try:
        cfg = app_config()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    examples = [
        json.loads(line)
        for line in Path(args.split).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    with read_only_connection(cfg) as conn:
        full_schema, _ = build_schema_context(conn)
        index = build_index(conn)
        retriever = KeywordRetriever(index, max_tables=args.max_tables,
                                     expand_neighbours=args.expand)

        print("=" * 72)
        print("SCHEMA RETRIEVAL — OFFLINE EVALUATION")
        print("=" * 72)
        print(f"examples          : {len(examples):,}")
        print(f"tables in schema  : {len(index.tables)}")
        print(f"value-indexed cols: {len(index.indexed_columns)}")
        print(f"full schema size  : {len(full_schema):,} chars")
        print(f"max seed tables   : {args.max_tables}\n")

        rows = []
        for ex in examples:
            gold = set(ex.get("referenced_tables") or [])
            result = retriever.retrieve(ex["question"])
            got = set(result.tables)
            rows.append({
                "id": ex["id"],
                "template_id": ex["template_id"],
                "difficulty": ex["difficulty"],
                "domain": ex["domain"],
                "question": ex["question"],
                "gold_tables": sorted(gold),
                "retrieved_tables": sorted(got),
                "bridge_tables": result.bridge_tables,
                "missing": sorted(gold - got),
                "extra": sorted(got - gold),
                "complete": gold.issubset(got),
                "n_retrieved": len(got),
                "schema_chars": len(result.schema_text),
                "matched_values": result.matched_values,
            })

    total_gold = sum(len(r["gold_tables"]) for r in rows)
    found_gold = sum(len(set(r["gold_tables"]) & set(r["retrieved_tables"]))
                     for r in rows)
    complete = sum(1 for r in rows if r["complete"])
    precision_vals = [
        len(set(r["gold_tables"]) & set(r["retrieved_tables"])) / max(r["n_retrieved"], 1)
        for r in rows
    ]
    sizes = [r["n_retrieved"] for r in rows]
    chars = [r["schema_chars"] for r in rows]

    print("RECALL (the metric that matters — a missing table makes the answer impossible)")
    print(f"  table recall (micro)   : {100.0 * found_gold / total_gold:.2f} %"
          f"  ({found_gold}/{total_gold} gold tables)")
    print(f"  complete retrieval     : {100.0 * complete / len(rows):.2f} %"
          f"  ({complete}/{len(rows)} questions got ALL needed tables)")
    print(f"  incomplete             : {len(rows) - complete}")

    print("\nPRECISION / COMPRESSION (secondary — extra tables cost tokens, not correctness)")
    print(f"  mean precision         : {100.0 * statistics.fmean(precision_vals):.2f} %")
    print(f"  tables retrieved       : mean {statistics.fmean(sizes):.2f} "
          f"| median {statistics.median(sizes):.0f} | max {max(sizes)}"
          f"  (of {len(index.tables)})")
    print(f"  schema size            : mean {statistics.fmean(chars):,.0f} chars "
          f"vs {len(full_schema):,} full "
          f"({100.0 * statistics.fmean(chars) / len(full_schema):.1f} %)")

    print("\nBY DIFFICULTY")
    by_diff: dict[str, list] = defaultdict(list)
    for r in rows:
        by_diff[r["difficulty"]].append(r)
    print(f"  {'tier':<12}{'n':>5}{'complete':>10}{'recall':>9}{'tables':>8}")
    for tier, group in sorted(by_diff.items()):
        c = sum(1 for r in group if r["complete"])
        g = sum(len(r["gold_tables"]) for r in group)
        f = sum(len(set(r["gold_tables"]) & set(r["retrieved_tables"])) for r in group)
        print(f"  {tier:<12}{len(group):>5}{100.0*c/len(group):>9.1f}%"
              f"{100.0*f/g:>8.1f}%{statistics.fmean([r['n_retrieved'] for r in group]):>8.2f}")

    misses = [r for r in rows if not r["complete"]]
    if misses:
        print(f"\nMISSED TABLES (top causes across {len(misses)} incomplete retrievals)")
        for table, n in Counter(t for r in misses for t in r["missing"]).most_common(8):
            print(f"  {table:<18}{n:>4}")
        print(f"\nEXAMPLES OF INCOMPLETE RETRIEVAL (first {args.show_misses})")
        for r in misses[:args.show_misses]:
            print(f"\n  [{r['template_id']}] {r['difficulty']}")
            print(f"  Q       : {r['question'][:88]}")
            print(f"  gold    : {r['gold_tables']}")
            print(f"  got     : {r['retrieved_tables']}")
            print(f"  MISSING : {r['missing']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail = OUTPUT_DIR / f"retrieval_eval_k{args.max_tables}e{args.expand}.jsonl"
    with detail.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "retriever": retriever.describe(),
        "examples": len(rows),
        "table_recall_pct": round(100.0 * found_gold / total_gold, 2),
        "complete_retrieval_pct": round(100.0 * complete / len(rows), 2),
        "incomplete_count": len(rows) - complete,
        "mean_precision_pct": round(100.0 * statistics.fmean(precision_vals), 2),
        "mean_tables_retrieved": round(statistics.fmean(sizes), 2),
        "total_tables": len(index.tables),
        "mean_schema_chars": round(statistics.fmean(chars), 1),
        "full_schema_chars": len(full_schema),
        "compression_pct": round(100.0 * statistics.fmean(chars) / len(full_schema), 1),
    }
    (OUTPUT_DIR / f"retrieval_summary_k{args.max_tables}e{args.expand}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nwritten to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
