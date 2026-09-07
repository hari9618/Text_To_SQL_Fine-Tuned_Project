"""Publish the LoRA adapter and its Model Card to the Hugging Face Hub.

Phase 14.

Uploads only what someone needs to *use* the adapter:

    adapter_model.safetensors   the trained weights (167 MB)
    adapter_config.json         LoRA rank, alpha, target modules
    tokenizer.json              vocabulary
    tokenizer_config.json       tokenizer settings
    chat_template.jinja         the template the prompt format depends on
    README.md                   the Model Card
    training_config.json        every hyperparameter, GPU, dtype, dataset stat
    training_metrics.json       the loss curve and final eval

The base model is *not* uploaded — it is 16 GB, it is already on the Hub, and
`adapter_config.json` points at the exact revision that was trained on.

Publishing is deliberately explicit: `--dry-run` is the default behaviour to
inspect, and `--private` is available, because a public push is hard to undo.

Usage (from the project root):

    # see exactly what would be uploaded, no network writes
    env\\Scripts\\python.exe scripts/upload_model.py --repo you/qwen3-8b-text2sql-qlora

    # actually publish
    env\\Scripts\\python.exe scripts/upload_model.py --repo you/... --push

Needs a token with **write** access:
<https://huggingface.co/settings/tokens> -> New token -> Write.
Put it in `.env` as `HF_WRITE_TOKEN` (kept separate from the read-only
`HF_TOKEN` used for inference, so an inference credential cannot publish).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sql.config import PROJECT_ROOT  # noqa: E402

ADAPTER_DIR = PROJECT_ROOT / "models" / "finetuned"
FINAL = ADAPTER_DIR / "final_adapter"
CARD = ADAPTER_DIR / "MODEL_CARD.md"

# (source, name in the repo). Anything not listed is not published.
FILES: list[tuple[Path, str]] = [
    (FINAL / "adapter_model.safetensors", "adapter_model.safetensors"),
    (FINAL / "adapter_config.json", "adapter_config.json"),
    (FINAL / "tokenizer.json", "tokenizer.json"),
    (FINAL / "tokenizer_config.json", "tokenizer_config.json"),
    (FINAL / "chat_template.jinja", "chat_template.jinja"),
    (ADAPTER_DIR / "training_config.json", "training_config.json"),
    (ADAPTER_DIR / "training_metrics.json", "training_metrics.json"),
    (CARD, "README.md"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish the adapter to the HF Hub")
    p.add_argument("--repo", required=True,
                   help="target repo id, e.g. yourname/qwen3-8b-text2sql-qlora")
    p.add_argument("--push", action="store_true",
                   help="actually upload; without it nothing is written")
    p.add_argument("--private", action="store_true",
                   help="create the repo private (can be made public later)")
    p.add_argument("--token-env", default="HF_WRITE_TOKEN",
                   help="environment variable holding a write-scoped token")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 74)
    print("PUBLISH ADAPTER TO THE HUGGING FACE HUB")
    print("=" * 74)
    print(f"repo      : {args.repo}{'  (private)' if args.private else '  (public)'}")
    print(f"mode      : {'PUSH' if args.push else 'DRY RUN - nothing will be written'}")
    print()

    missing = [str(src.relative_to(PROJECT_ROOT)) for src, _ in FILES
               if not src.exists()]
    if missing:
        print("[error] missing files:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        if any("MODEL_CARD" in m for m in missing):
            print("\n  The Model Card is required. A model published without one "
                  "is\n  much less useful to anyone who finds it.", file=sys.stderr)
        return 2

    total = 0
    print("files to publish:")
    for src, dest in FILES:
        size = src.stat().st_size
        total += size
        print(f"  {size / 1e6:9.2f} MB  {dest}")
    print(f"  {'-' * 9}")
    print(f"  {total / 1e6:9.2f} MB  total")
    print()

    # The base weights must never be uploaded: 16 GB already on the Hub.
    assert not any("model-0000" in d for _, d in FILES), "base weights included"

    card = CARD.read_text(encoding="utf-8")
    if "<your-username>" in card:
        card = card.replace("<your-username>/qwen3-8b-text2sql-qlora", args.repo)
        print(f"Model Card: substituted the repo id into the usage example")
    if not card.startswith("---"):
        print("[warning] the Model Card has no YAML frontmatter; the Hub will "
              "not show base_model or tags")

    if not args.push:
        print()
        print("Dry run. Re-run with --push to publish.")
        print(f"A write-scoped token must be in the environment as "
              f"{args.token_env}.")
        return 0

    token = os.getenv(args.token_env, "").strip()
    if not token:
        print(f"\n[error] {args.token_env} is not set.\n"
              f"        Create a token with WRITE access at\n"
              f"        https://huggingface.co/settings/tokens\n"
              f"        and add it to .env as {args.token_env}=hf_...",
              file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        who = api.whoami()
    except Exception as exc:  # noqa: BLE001
        print(f"[error] token rejected: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(f"authenticated as {who.get('name')}")

    owner = args.repo.split("/")[0]
    if who.get("name") != owner and owner not in [
            o.get("name") for o in who.get("orgs", [])]:
        print(f"[error] token belongs to {who.get('name')!r} but the repo is "
              f"owned by {owner!r}.\n"
              f"        Publishing would fail, or land in the wrong account.",
              file=sys.stderr)
        return 2

    api.create_repo(args.repo, repo_type="model", private=args.private,
                    exist_ok=True)
    print(f"repo ready: https://huggingface.co/{args.repo}")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "README.md"
        staged.write_text(card, encoding="utf-8")
        for src, dest in FILES:
            source = staged if dest == "README.md" else src
            print(f"  uploading {dest} ...", flush=True)
            api.upload_file(path_or_fileobj=str(source), path_in_repo=dest,
                            repo_id=args.repo, repo_type="model")

    print()
    print("=" * 74)
    print(f"published: https://huggingface.co/{args.repo}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
