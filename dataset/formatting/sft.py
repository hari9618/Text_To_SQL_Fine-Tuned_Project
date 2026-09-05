"""Build supervised fine-tuning records from validated benchmark examples.

Phase 7 (CLAUDE.md numbering).

The training task is exactly the task the baseline was measured on:

    system instruction + full schema + question   ->   PostgreSQL SQL

**The prompt is imported from `src/model/prompt.py`, never re-written here.**
That is the single most important decision in this module. Phase 10 compares a
fine-tuned model against the frozen Phase 5 baseline (10.82 % strict execution
accuracy). If training used a different prompt from evaluation, the measured
difference would mix fine-tuning with prompt engineering and neither could be
attributed. Reusing the module guarantees they cannot drift apart: change the
prompt and both the baseline and this dataset are invalidated together, loudly.

**Full schema, not retrieved.** Phase 6 measured schema retrieval on this
12-table database and it *reduced* strict accuracy from 10.82 % to 9.27 %. The
full schema is therefore the pipeline default, and training on it keeps the
fine-tuned model comparable to the frozen baseline.

## What the model must not see

Validated examples carry a great deal of bookkeeping:

    template_id        which template generated the question
    phrasing_index     which paraphrase
    slot_values        the values substituted into the template
    execution          gold result fingerprint, row count, latency
    referenced_tables  the tables the gold query touches
    split              train / validation / test
    difficulty, domain, query_type

None of it belongs in the prompt. Some would be outright leakage —
`referenced_tables` hands over the schema-linking half of the problem, and
`execution.fingerprint` is derived from the answer itself. The rest would teach
the model to expect labels that will not exist at inference time.

These fields are preserved in a `meta` object *outside* the messages, so
per-difficulty training analysis remains possible without any of it reaching
the model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.model.prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_messages,
    prompt_fingerprint,
)

# Fields that exist only for evaluation and analysis. Never rendered into a
# prompt; carried in `meta` for offline slicing.
METADATA_FIELDS = (
    "template_id", "phrasing_index", "slot_values", "execution",
    "referenced_tables", "split", "difficulty", "domain", "query_type", "id",
)

# Substrings that must never appear in a rendered prompt. Checked mechanically
# so a future refactor cannot quietly reintroduce leakage.
FORBIDDEN_IN_PROMPT = (
    "template_id", "phrasing_index", "slot_values", "referenced_tables",
    "fingerprint", "row_count", '"split"', "difficulty", "query_type",
)


@dataclass
class SFTRecord:
    """One training example in chat format."""

    example_id: str
    messages: list[dict[str, str]]
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "messages": self.messages,
            "meta": self.meta,
        }

    @property
    def prompt_text(self) -> str:
        """Everything the model reads, for leakage and length checks."""
        return "\n".join(
            m["content"] for m in self.messages if m["role"] != "assistant"
        )

    @property
    def completion(self) -> str:
        return self.messages[-1]["content"]


def normalise_sql(sql: str) -> str:
    """Collapse whitespace. The gold SQL is already single-line from Phase 2,
    but this makes the assistant turn deterministic regardless of source."""
    return " ".join(sql.split()).strip()


def build_record(example: dict[str, Any], schema_text: str) -> SFTRecord:
    """Convert one validated example into a chat-format training record.

    The assistant turn is the gold SQL and nothing else — no prose, no markdown
    fence, no trailing semicolon. The baseline harness already strips fences and
    commentary when reading model output, so training the model to emit bare SQL
    removes work the extractor would otherwise have to do, and removes a way for
    the model to waste tokens on explanation it was told not to give.
    """
    messages = build_messages(example["question"], schema_text)
    messages.append({"role": "assistant", "content": normalise_sql(example["sql"])})

    meta = {k: example[k] for k in METADATA_FIELDS if k in example and k != "id"}
    # The gold fingerprint is useful for offline checks but must not travel
    # inside `execution`, which also carries latency noise from Phase 3.
    execution = meta.pop("execution", None)
    if isinstance(execution, dict):
        meta["gold_fingerprint"] = execution.get("fingerprint")
        meta["gold_row_count"] = execution.get("row_count")

    return SFTRecord(example_id=example["id"], messages=messages, meta=meta)


def build_records(
    examples: Iterable[dict[str, Any]], schema_text: str
) -> list[SFTRecord]:
    return [build_record(e, schema_text) for e in examples]


def file_hash(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def content_hash(records: list[SFTRecord]) -> str:
    """Hash of the rendered records, independent of file formatting.

    Two runs producing the same conversations hash identically even if the JSON
    serialiser changes, which is what makes "the training data did not change"
    a checkable claim.
    """
    digest = hashlib.sha256()
    for record in records:
        for message in record.messages:
            digest.update(message["role"].encode())
            digest.update(b"\x00")
            digest.update(message["content"].encode())
            digest.update(b"\x01")
    return digest.hexdigest()[:16]


def describe_format() -> dict[str, Any]:
    """Recorded alongside the dataset so training can verify what it received."""
    return {
        "task": "question + full schema -> PostgreSQL SQL",
        "format": "chat messages (system, user, assistant)",
        "prompt_version": PROMPT_VERSION,
        "prompt_fingerprint": prompt_fingerprint(),
        "prompt_source": "src/model/prompt.py (shared with the Phase 5 baseline)",
        "schema_mode": "full_schema_no_retrieval",
        "schema_rationale": (
            "Phase 6 measured retrieval on this 12-table schema: strict accuracy "
            "fell from 10.82 % to 9.27 %. Full schema is the pipeline default."
        ),
        "assistant_content": "gold SQL only, whitespace-normalised, no fence, no prose",
        "thinking_mode": "disabled via /no_think, matching baseline inference",
        "system_prompt_chars": len(SYSTEM_PROMPT),
    }
