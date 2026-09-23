"""Publish the Text-to-SQL benchmark to the Hugging Face Hub.

Phase 14.

Publishes the **benchmark** rendering (question, gold SQL, difficulty, execution
fingerprint) rather than the SFT chat rendering. The SFT files embed the full
5,455-character schema in every one of 2,133 records — 14 MB, of which ~12 MB is
the same schema repeated — and are exactly reproducible from what is published
here plus `scripts/prepare_sft_dataset.py`. Shipping the redundancy would make
the dataset harder to use, not easier.

Also ships the database DDL and the rendered schema context, because gold SQL
without a schema is not reproducible: you cannot execute it, and execution is
how this benchmark defines correctness.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/upload_dataset.py --repo you/name
    env\\Scripts\\python.exe scripts/upload_dataset.py --repo you/name --push

Needs `HF_WRITE_TOKEN` in the environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_versions import add_version_argument, get as get_version
from src.sql.config import PROJECT_ROOT  # noqa: E402



def lf(data: bytes) -> bytes:
    """Normalise CRLF to LF before hashing or publishing.

    The working copy has Windows line endings. Publishing those would give
    anyone who downloads a file a different hash than the fingerprint this
    benchmark documents, which is exactly the drift the fingerprints exist
    to catch.
    """
    return data.replace(b"\r\n", b"\n")

def dataset_files(version) -> tuple[Path, Path, list[tuple[Path, str]]]:
    """This version's splits, plus the schema the questions were written
    against and the rendered schema context the model is shown."""
    card = PROJECT_ROOT / "dataset" / "DATASET_CARD.md"
    schema_ctx = version.experiments_root / "finetuned" / "evalpack" / "schema_context.txt"
    if not schema_ctx.exists():   # the eval pack is the canonical rendering
        schema_ctx = PROJECT_ROOT / "experiments/finetuned/evalpack/schema_context.txt"
    return card, schema_ctx, [
        (version.split("train"), "train.jsonl"),
        (version.split("validation"), "validation.jsonl"),
        (version.split("test"), "test.jsonl"),
        (PROJECT_ROOT / "database/schema.sql", "schema.sql"),
        (schema_ctx, "schema_context.txt"),
        (card, "README.md"),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish the benchmark dataset")
    add_version_argument(p)
    p.add_argument("--repo", required=True)
    p.add_argument("--push", action="store_true")
    p.add_argument("--private", action="store_true")
    p.add_argument("--token-env", default="HF_WRITE_TOKEN")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    version = get_version(args.version)
    CARD, SCHEMA_CTX, FILES = dataset_files(version)

    print("=" * 74)
    print(f"PUBLISH BENCHMARK DATASET  (benchmark {version.name})")
    print("=" * 74)
    print(f"repo : {args.repo}{'  (private)' if args.private else '  (public)'}")
    print(f"mode : {'PUSH' if args.push else 'DRY RUN - nothing will be written'}")
    print()

    missing = [str(s.relative_to(PROJECT_ROOT)) for s, _ in FILES if not s.exists()]
    if missing:
        print("[error] missing:", *missing, sep="\n  ", file=sys.stderr)
        return 2

    # ---- integrity, before anything is published --------------------------
    splits = {}
    for name in ("train", "validation", "test"):
        path = version.split(name)
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        splits[name] = rows

    ids = {k: {r["id"] for r in v} for k, v in splits.items()}
    templates = {k: {r["template_id"] for r in v} for k, v in splits.items()}

    checks = [
        ("splits are disjoint by example id",
         not (ids["train"] & ids["test"] or ids["train"] & ids["validation"]
              or ids["validation"] & ids["test"])),
        ("train and test share no template",
         not (templates["train"] & templates["test"])),
        ("validation and test share no template",
         not (templates["validation"] & templates["test"])),
        ("train and validation share no template",
         not (templates["train"] & templates["validation"])),
        ("every row carries gold SQL",
         all(r.get("sql") for v in splits.values() for r in v)),
        ("every row records its split",
         all(r.get("split") == k for k, v in splits.items() for r in v)),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not all(ok for _, ok in checks):
        print("\n[abort] the dataset does not satisfy its own claims.",
              file=sys.stderr)
        return 1

    print()
    print(f"  {'split':<14}{'rows':>7}{'templates':>11}")
    for name, rows in splits.items():
        print(f"  {name:<14}{len(rows):>7}{len(templates[name]):>11}")
    print(f"  {'total':<14}{sum(len(v) for v in splits.values()):>7}")

    # Hash the LF-normalised bytes. The working copy has CRLF endings on
    # Windows, and publishing those would give anyone who downloads the file a
    # different hash than the fingerprint this benchmark documents. Every text
    # file is normalised on upload for the same reason.
    schema_fp = hashlib.sha256(lf(SCHEMA_CTX.read_bytes())).hexdigest()[:16]
    print(f"\n  schema fingerprint: {schema_fp}")
    if schema_fp != "d03619e711661bc5":
        print("[abort] schema drifted from the frozen benchmark.", file=sys.stderr)
        return 1

    print()
    total = 0
    for src, dest in FILES:
        total += src.stat().st_size
        print(f"  {src.stat().st_size / 1024:9,.0f} KB  {dest}")
    print(f"  {'-' * 9}")
    print(f"  {total / 1024:9,.0f} KB  total")

    card = CARD.read_text(encoding="utf-8").replace("<repo-id>", args.repo)

    if not args.push:
        print("\nDry run. Re-run with --push to publish.")
        return 0

    token = os.getenv(args.token_env, "").strip()
    if not token:
        print(f"\n[error] {args.token_env} is not set.", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami()
    owner = args.repo.split("/")[0]
    if who.get("name") != owner:
        print(f"[error] token belongs to {who.get('name')!r}, repo owner is "
              f"{owner!r}", file=sys.stderr)
        return 2

    api.create_repo(args.repo, repo_type="dataset", private=args.private,
                    exist_ok=True)
    print(f"\nrepo ready: https://huggingface.co/datasets/{args.repo}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for src, dest in FILES:
            staged = tmpdir / dest
            body = card.encode("utf-8") if dest == "README.md" else src.read_bytes()
            # Publish LF endings so hashes are stable across platforms.
            staged.write_bytes(lf(body))
            print(f"  uploading {dest} ...", flush=True)
            api.upload_file(path_or_fileobj=str(staged), path_in_repo=dest,
                            repo_id=args.repo, repo_type="dataset")

    print()
    print("=" * 74)
    print(f"published: https://huggingface.co/datasets/{args.repo}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
