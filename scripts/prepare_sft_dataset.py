"""Build the supervised fine-tuning dataset. No training happens here.

Phase 7 (CLAUDE.md numbering).

    dataset/train/train.jsonl            ->  dataset/sft/train.jsonl
    dataset/validation/validation.jsonl  ->  dataset/sft/validation.jsonl
    dataset/test/test.jsonl              ->  never read for content

Every check below is a guard against a specific way a fine-tuning dataset can
look fine and silently invalidate the experiment:

* **Test contamination** — a question or gold query appearing in both training
  and test makes the Phase 10 score memorisation rather than generalisation.
  Checked at question level *and* SQL level, not just by split label.
* **Metadata leakage** — `referenced_tables` would hand the model the
  schema-linking half of the task; `execution.fingerprint` is derived from the
  answer. Prompts are scanned mechanically for both.
* **Prompt drift** — training must use the exact prompt the frozen baseline was
  measured with, or Phase 10 measures prompt engineering instead of
  fine-tuning. The template is imported, and its fingerprint is asserted
  against the baseline run.
* **Schema drift** — the schema is rendered once and its hash compared to the
  one recorded in the baseline run.
* **Duplicates** — identical question/SQL pairs inflate the apparent dataset
  size and over-weight whatever they teach.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/prepare_sft_dataset.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.formatting.sft import (  # noqa: E402
    FORBIDDEN_IN_PROMPT,
    SFTRecord,
    build_records,
    content_hash,
    describe_format,
    file_hash,
    normalise_sql,
)
from src.benchmark_versions import add_version_argument, get as get_version  # noqa: E402
from src.model.schema_context import build_schema_context  # noqa: E402
from src.sql.config import PROJECT_ROOT, ConfigError, app_config  # noqa: E402
from src.sql.executor import read_only_connection  # noqa: E402


# Recorded when the Phase 5 baseline was frozen. Training must match.
BASELINE_PROMPT_FINGERPRINT = "8288e41a496531a9"
BASELINE_SCHEMA_FINGERPRINT = "d03619e711661bc5"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


class CheckFailed(Exception):
    pass


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        raise CheckFailed(f"{label}: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the SFT dataset")
    add_version_argument(parser)
    args = parser.parse_args()
    version = get_version(args.version)
    prompt = version.prompt()
    prompt_fingerprint = prompt.prompt_fingerprint
    SOURCES = {name: version.split(name) for name in ("train", "validation")}
    TEST_SET = version.split("test")
    OUT_DIR = version.sft_dir
    # v1 must match the frozen baseline; later versions match their own prompt module.
    expected_prompt_fp = BASELINE_PROMPT_FINGERPRINT if version.is_v1 else prompt.prompt_fingerprint()

    try:
        cfg = app_config()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    print("=" * 72)
    print("PHASE 7 — SUPERVISED FINE-TUNING DATASET PREPARATION")
    print("=" * 72)

    with read_only_connection(cfg) as conn:
        schema_text, schema_fp = build_schema_context(conn)

    print(f"schema      : full schema, {len(schema_text):,} chars (hash {schema_fp})")
    print(f"benchmark   : {version.name}")
    print(f"prompt      : {prompt_fingerprint()} ({prompt.PROMPT_VERSION}, {version.prompt_module})\n")

    print("--- consistency with the frozen baseline ---")
    check(prompt_fingerprint() == expected_prompt_fp,
          "prompt template matches the frozen Phase 5 baseline" if version.is_v1
          else f"prompt template matches {version.prompt_module}",
          f"got {prompt_fingerprint()}, expected {expected_prompt_fp}")
    check(schema_fp == BASELINE_SCHEMA_FINGERPRINT,
          "rendered schema matches the frozen Phase 5 baseline",
          f"got {schema_fp}, expected {BASELINE_SCHEMA_FINGERPRINT}")

    # Test set is read for contamination checking only; its content never
    # informs any formatting decision and it is never written out.
    test_examples = load(TEST_SET)
    test_questions = {e["question"] for e in test_examples}
    test_sql = {normalise_sql(e["sql"]) for e in test_examples}
    test_templates = {e["template_id"] for e in test_examples}
    print(f"\n--- test set loaded for contamination checks only "
          f"({len(test_examples)} examples, never written) ---")

    splits: dict[str, list[SFTRecord]] = {}
    stats: dict[str, dict] = {}

    for name, path in SOURCES.items():
        examples = load(path)
        print(f"\n--- {name}: {len(examples):,} source examples ---")

        # ---- completeness -------------------------------------------------
        missing = [e.get("id") for e in examples
                   if not e.get("question", "").strip() or not e.get("sql", "").strip()]
        check(not missing, "every example has a question and gold SQL",
              f"{len(missing)} incomplete")

        records = build_records(examples, schema_text, prompt)

        check(all(len(r.messages) == 3 for r in records),
              "every record has system + user + assistant turns")
        check(all(r.messages[0]["role"] == "system"
                  and r.messages[1]["role"] == "user"
                  and r.messages[2]["role"] == "assistant" for r in records),
              "message roles are in the required order")

        # ---- the assistant turn is exactly the validated gold SQL ---------
        by_id = {e["id"]: e for e in examples}
        mismatched = [r.example_id for r in records
                      if r.completion != normalise_sql(by_id[r.example_id]["sql"])]
        check(not mismatched, "assistant turn is exactly the validated gold SQL",
              f"{len(mismatched)} differ")

        starts_ok = [r for r in records
                     if not r.completion.upper().startswith(("SELECT", "WITH"))]
        check(not starts_ok, "every completion starts with SELECT or WITH",
              f"{len(starts_ok)} do not")
        fenced = [r for r in records if "```" in r.completion or "--" in r.completion]
        check(not fenced, "no completion contains a markdown fence or comment",
              f"{len(fenced)} do")

        # ---- the schema is present, whole, and identical everywhere -------
        check(all(schema_text in r.messages[1]["content"] for r in records),
              "full schema is embedded in every user turn")
        check(len({r.messages[0]["content"] for r in records}) == 1,
              "system instruction is identical across all records")

        # ---- no metadata reaches the model --------------------------------
        leaks: list[str] = []
        for r in records[:]:
            lowered = r.prompt_text.lower()
            for term in FORBIDDEN_IN_PROMPT:
                if term.lower() in lowered:
                    leaks.append(f"{r.example_id}:{term}")
        check(not leaks, "no evaluation metadata appears in any prompt",
              f"{len(leaks)} leaks, e.g. {leaks[:3]}")

        # `id` is metadata too: it must not be inside the conversation.
        id_leaks = [r.example_id for r in records if r.example_id in r.prompt_text]
        check(not id_leaks, "example ids do not appear in prompts",
              f"{len(id_leaks)} leaked")

        # ---- duplicates ---------------------------------------------------
        pairs = Counter((r.messages[1]["content"], r.completion) for r in records)
        dupes = {k: v for k, v in pairs.items() if v > 1}
        check(not dupes, "no duplicate question/SQL pairs within the split",
              f"{len(dupes)} duplicated")

        questions = Counter(e["question"] for e in examples)
        qdupes = {q: n for q, n in questions.items() if n > 1}
        check(not qdupes, "no duplicate questions within the split",
              f"{len(qdupes)} duplicated")

        # ---- test contamination -------------------------------------------
        q_overlap = sorted({e["question"] for e in examples} & test_questions)
        check(not q_overlap, "no question appears in the test set",
              f"{len(q_overlap)} shared, e.g. {q_overlap[:2]}")

        s_overlap = sorted({normalise_sql(e["sql"]) for e in examples} & test_sql)
        check(not s_overlap, "no gold SQL appears in the test set",
              f"{len(s_overlap)} shared")

        t_overlap = sorted({e["template_id"] for e in examples} & test_templates)
        check(not t_overlap, "no template is shared with the test set",
              f"{len(t_overlap)} shared: {t_overlap[:5]}")

        splits[name] = records
        stats[name] = {
            "source_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "source_sha256_16": file_hash(path),
            "examples": len(records),
            "templates": len({e["template_id"] for e in examples}),
            "by_difficulty": dict(Counter(e["difficulty"] for e in examples).most_common()),
            "by_domain": dict(Counter(e["domain"] for e in examples).most_common()),
        }

    # ---- train/validation must not overlap either -------------------------
    print("\n--- train vs validation ---")
    tr_q = {r.messages[1]["content"] for r in splits["train"]}
    va_q = {r.messages[1]["content"] for r in splits["validation"]}
    check(not (tr_q & va_q), "no prompt shared between train and validation",
          f"{len(tr_q & va_q)} shared")

    tr_t = set(stats["train"]["by_difficulty"])  # placeholder to keep symmetry
    train_templates = {r.meta["template_id"] for r in splits["train"]}
    val_templates = {r.meta["template_id"] for r in splits["validation"]}
    check(not (train_templates & val_templates),
          "no template shared between train and validation",
          f"{len(train_templates & val_templates)} shared")

    # ---- token statistics, for choosing max_seq_len in Phase 8 ------------
    print("\n--- token statistics (Qwen3 tokenizer) ---")
    try:
        import os

        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(
            hf_hub_download("Qwen/Qwen3-8B", "tokenizer.json",
                            token=os.getenv("HF_TOKEN"))
        )

        def count(text: str) -> int:
            return len(tok.encode(text).ids)

        for name, records in splits.items():
            prompt_tokens = [count(r.prompt_text) for r in records]
            completion_tokens = [count(r.completion) for r in records]
            total = [p + c for p, c in zip(prompt_tokens, completion_tokens)]
            stats[name]["tokens"] = {
                "prompt_mean": round(statistics.fmean(prompt_tokens), 1),
                "prompt_max": max(prompt_tokens),
                "completion_mean": round(statistics.fmean(completion_tokens), 1),
                "completion_max": max(completion_tokens),
                "total_mean": round(statistics.fmean(total), 1),
                "total_max": max(total),
                "total_p99": sorted(total)[int(0.99 * len(total))],
                "note": "excludes chat-template control tokens added at training time",
            }
            t = stats[name]["tokens"]
            print(f"  {name:<11} prompt mean {t['prompt_mean']:>7.0f} | "
                  f"completion mean {t['completion_mean']:>5.0f} | "
                  f"total max {t['total_max']:>5} | p99 {t['total_p99']:>5}")
        longest = max(s["tokens"]["total_max"] for s in stats.values())
        suggested = 1024 if longest <= 960 else (2048 if longest <= 1960 else 4096)
        print(f"  -> suggested max_seq_len for Phase 8: {suggested} "
              f"(longest example {longest} tokens)")
    except Exception as exc:  # tokenizer unavailable offline
        print(f"  [skipped] {type(exc).__name__}: {exc}")
        suggested = None

    # ---- write -------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n--- writing ---")
    for name, records in splits.items():
        out = OUT_DIR / f"{name}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
        stats[name]["output_path"] = str(out.relative_to(PROJECT_ROOT)).replace("\\", "/")
        stats[name]["output_sha256_16"] = file_hash(out)
        stats[name]["content_hash"] = content_hash(records)
        print(f"  {out.relative_to(PROJECT_ROOT)}  "
              f"{len(records):,} records  "
              f"file {stats[name]['output_sha256_16']}  "
              f"content {stats[name]['content_hash']}")

    manifest = {
        "phase": 7,
        "purpose": "supervised fine-tuning dataset for Qwen3-8B",
        "training_not_started": True,
        "format": describe_format(prompt),
        "schema": {
            "mode": "full_schema_no_retrieval",
            "fingerprint": schema_fp,
            "chars": len(schema_text),
            "matches_frozen_baseline": schema_fp == BASELINE_SCHEMA_FINGERPRINT,
        },
        "excluded_from_prompt": list(FORBIDDEN_IN_PROMPT),
        "test_set": {
            "path": "dataset/test/test.jsonl",
            "sha256_16": file_hash(TEST_SET),
            "examples": len(test_examples),
            "used_for": "contamination checking only; never written, never trained on",
        },
        "suggested_max_seq_len": suggested,
        "splits": stats,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    print(f"  {(OUT_DIR / 'manifest.json').relative_to(PROJECT_ROOT)}")

    # ---- human-readable sample --------------------------------------------
    sample = splits["train"][0]
    lines = [
        "# SFT record format — one worked example",
        "",
        "Produced by `scripts/prepare_sft_dataset.py`. This is exactly what the",
        "model sees during Phase 9 training: three chat turns, nothing else.",
        "The `meta` block below sits *outside* `messages` and is never rendered.",
        "",
        f"- prompt template: `{describe_format(prompt)['prompt_version']}` "
        f"(fingerprint `{prompt_fingerprint()}`) — "
        + ("identical to the frozen baseline" if version.is_v1 else f"from {version.prompt_module}"),
        f"- schema: full, {len(schema_text):,} chars, fingerprint `{schema_fp}`",
        "",
        "---",
        "",
        "## 1. system",
        "",
        "```text",
        sample.messages[0]["content"],
        "```",
        "",
        "## 2. user",
        "",
        "The full schema is embedded here. Truncated below for readability —",
        f"the real record contains all {len(schema_text):,} characters.",
        "",
        "```text",
        sample.messages[1]["content"][:700],
        "",
        "        ... schema continues ...",
        "",
        sample.messages[1]["content"][-320:],
        "```",
        "",
        "## 3. assistant",
        "",
        "SQL only — no prose, no markdown fence, no trailing semicolon.",
        "",
        "```sql",
        sample.completion,
        "```",
        "",
        "---",
        "",
        "## meta (not shown to the model)",
        "",
        "```json",
        json.dumps(sample.meta, indent=2),
        "```",
        "",
        "Carried for offline analysis only — per-difficulty training curves, "
        "error slicing. None of it reaches the prompt; "
        "`scripts/prepare_sft_dataset.py` asserts that mechanically.",
    ]
    (OUT_DIR / "SAMPLE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  {(OUT_DIR / 'SAMPLE.md').relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 72)
    print(f"train      {stats['train']['examples']:>6,} records")
    print(f"validation {stats['validation']['examples']:>6,} records")
    print(f"test       {len(test_examples):>6,} records  (UNTOUCHED)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckFailed as exc:
        print(f"\n[abort] integrity check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
