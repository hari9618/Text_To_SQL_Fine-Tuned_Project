"""Export the inputs a GPU host needs to generate Phase 10 predictions.

Phase 10, step 1 of 3.

Generation needs a GPU. Scoring needs PostgreSQL. Those live on different
machines, so Phase 10 is split:

    this script  ->  evalpack.zip  ->  Kaggle notebook  ->  predictions.jsonl
                                                                   |
                                            scripts/score_finetuned.py
                                                                   |
                                                                 verdict

What travels to the GPU is deliberately minimal:

* **questions only, never the gold SQL.** The generator cannot copy an answer
  it was never given. This is a structural guarantee, not a promise.
* **the frozen schema text**, rendered from the live database here and hashed,
  so the fine-tuned model sees byte-identical context to the frozen baseline.
* **`src/model/prompt.py` verbatim**, so the remote host cannot paraphrase the
  prompt. A reworded prompt would make Phase 10 measure prompt engineering
  rather than fine-tuning.

Both fingerprints are asserted against the values frozen in Phase 4/7. If
either has drifted the export aborts, because a comparison against the 10.82 %
baseline would no longer be like for like.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/export_eval_pack.py
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

from src.evaluation.baseline import dataset_fingerprint  # noqa: E402
from src.model.prompt import PROMPT_VERSION, prompt_fingerprint  # noqa: E402
from src.model.schema_context import build_schema_context  # noqa: E402
from src.sql.config import PROJECT_ROOT, ConfigError, app_config  # noqa: E402
from src.sql.executor import read_only_connection  # noqa: E402

# Frozen in Phase 4 (baseline) and re-asserted in Phase 7 (SFT dataset).
# Changing either invalidates both, deliberately.
EXPECTED_PROMPT_FP = "8288e41a496531a9"
EXPECTED_SCHEMA_FP = "d03619e711661bc5"

TEST_SET = PROJECT_ROOT / "dataset" / "test" / "test.jsonl"
PROMPT_SRC = PROJECT_ROOT / "src" / "model" / "prompt.py"
OUT_DIR = PROJECT_ROOT / "experiments" / "finetuned" / "evalpack"


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 10 eval pack export")
    p.add_argument("--retrieval", choices=["none", "keyword"], default="none",
                   help="'none' ships the full schema (ablation configuration "
                        "3); 'keyword' ships a retrieved subset per question "
                        "(configuration 4)")
    p.add_argument("--retrieval-k", type=int, default=4,
                   help="max seed tables (tuned on validation in Phase 6)")
    p.add_argument("--retrieval-expand", type=int, default=2,
                   help="FK neighbour expansion (tuned on validation)")
    p.add_argument("--zip-to", default=None,
                   help="where to write the uploadable zip")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not TEST_SET.exists():
        print(f"[error] test split not found: {TEST_SET}", file=sys.stderr)
        return 2

    try:
        cfg = app_config()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    print("=" * 72)
    print("PHASE 10 - EXPORT EVAL PACK")
    print("=" * 72)

    # ---- prompt -----------------------------------------------------------
    prompt_fp = prompt_fingerprint()
    if prompt_fp != EXPECTED_PROMPT_FP:
        print(f"[abort] prompt fingerprint drifted: {prompt_fp} "
              f"!= {EXPECTED_PROMPT_FP}\n"
              f"        The frozen baseline used {EXPECTED_PROMPT_FP}. Comparing "
              f"against it now would measure the prompt change, not the "
              f"fine-tune.", file=sys.stderr)
        return 2
    print(f"prompt          : {PROMPT_VERSION}  {prompt_fp}  OK")

    # ---- schema, rendered from the live database --------------------------
    with read_only_connection(cfg) as conn:
        schema_text, schema_fp = build_schema_context(conn)

    if schema_fp != EXPECTED_SCHEMA_FP:
        print(f"[abort] schema fingerprint drifted: {schema_fp} "
              f"!= {EXPECTED_SCHEMA_FP}\n"
              f"        The database no longer renders the schema the baseline "
              f"saw.", file=sys.stderr)
        return 2
    print(f"schema          : {schema_fp}  {len(schema_text):,} chars  OK")

    # ---- questions, without answers ---------------------------------------
    with TEST_SET.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    questions = [{"id": r["id"], "question": r["question"]} for r in rows]

    # Configuration 4: each question sees only the tables a retriever selected.
    # The retriever runs here, on the machine with the database, so the GPU host
    # receives text and never a connection.
    retrieval_meta: dict | None = None
    if args.retrieval == "keyword":
        with read_only_connection(cfg) as conn:
            from src.retrieval.keyword import KeywordRetriever
            from src.retrieval.schema_index import build_index

            retriever = KeywordRetriever(
                build_index(conn), max_tables=args.retrieval_k,
                expand_neighbours=args.retrieval_expand)
            sizes, table_counts = [], []
            for q in questions:
                r = retriever.retrieve(q["question"])
                q["schema"] = r.schema_text
                sizes.append(len(r.schema_text))
                table_counts.append(len(r.tables))
        retrieval_meta = {
            "retriever": retriever.name,
            "max_tables": args.retrieval_k,
            "expand_neighbours": args.retrieval_expand,
            "mean_schema_chars": round(sum(sizes) / len(sizes), 1),
            "mean_tables_shown": round(sum(table_counts) / len(table_counts), 2),
            "full_schema_chars": len(schema_text),
            "tuned_on": "validation split, never test",
        }
        print(f"retrieval       : {retriever.name} k={args.retrieval_k} "
              f"expand={args.retrieval_expand}")
        print(f"                  mean {retrieval_meta['mean_schema_chars']:,.0f} "
              f"chars / {retrieval_meta['mean_tables_shown']:.2f} tables "
              f"(full schema {len(schema_text):,})")

    allowed = {"id", "question"} | ({"schema"} if args.retrieval != "none" else set())
    # Belt and braces: prove nothing but the allowed fields is leaving.
    leaked = {k for q in questions for k in q} - allowed
    assert not leaked, f"eval pack would leak fields: {sorted(leaked)}"
    assert all("SELECT" not in q["question"].upper() for q in questions), \
        "a question appears to contain SQL"

    questions_blob = "\n".join(
        json.dumps(q, ensure_ascii=False, sort_keys=True) for q in questions
    )
    questions_fp = sha16(questions_blob)

    # The notebook recomputes this exact hash over (id, schema) pairs and
    # aborts on a mismatch, so a truncated or edited pack cannot be trained on.
    if args.retrieval != "none":
        combined = hashlib.sha256()
        for q in questions:
            combined.update(q["id"].encode()); combined.update(bytes([0]))
            combined.update(q["schema"].encode()); combined.update(bytes([1]))
        shipped_schema_fp = combined.hexdigest()[:16]
    else:
        shipped_schema_fp = schema_fp
    print(f"questions       : {len(questions):,}  {questions_fp}  "
          f"(gold SQL withheld)")

    # ---- write ------------------------------------------------------------
    out_dir = (OUT_DIR if args.retrieval == "none"
               else OUT_DIR.parent / "evalpack_retrieved")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    (out_dir / "eval_questions.jsonl").write_text(
        questions_blob + "\n", encoding="utf-8")
    (out_dir / "schema_context.txt").write_text(schema_text, encoding="utf-8")
    shutil.copyfile(PROMPT_SRC, out_dir / "prompt_module.py")

    manifest = {
        "phase": 10,
        "purpose": "inputs for remote generation; scoring happens locally",
        "prompt": {"version": PROMPT_VERSION, "fingerprint": prompt_fp},
        "schema": {
            "mode": "full" if args.retrieval == "none" else "retrieved",
            "fingerprint": shipped_schema_fp,
            "full_schema_fingerprint": schema_fp,
            "characters": len(schema_text),
            "retrieval": retrieval_meta,
        },
        "ablation_configuration": 3 if args.retrieval == "none" else 4,
        "questions": {
            "count": len(questions),
            "fingerprint": questions_fp,
            "source": "dataset/test/test.jsonl",
            "source_fingerprint": dataset_fingerprint(TEST_SET),
            "gold_sql_included": False,
        },
        "baseline_to_beat": {
            "strict_execution_accuracy_pct": 10.82,
            "projection_tolerant_accuracy_pct": 45.92,
            "executable_sql_pct": 98.90,
            "schema_hallucination_pct": 0.66,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    # ---- zip for upload ---------------------------------------------------
    default_zip = ("text2sql-evalpack.zip" if args.retrieval == "none"
                   else "text2sql-evalpack-retrieved.zip")
    zip_path = Path(args.zip_to or (Path.home() / "Downloads" / default_zip))
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out_dir.iterdir()):
            zf.write(f, f.name)

    print("-" * 72)
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.stat().st_size / 1024:8.1f} KB  "
              f"{f.relative_to(PROJECT_ROOT)}")
    print("-" * 72)
    print(f"upload this: {zip_path}  ({zip_path.stat().st_size / 1024:.0f} KB)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
