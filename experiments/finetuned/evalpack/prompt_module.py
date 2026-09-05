"""The prompt template used to ask a model for SQL.

Phase 4.

Versioned and hashed on purpose. The prompt is as much a part of the
experiment as the model weights: a better-worded prompt raises accuracy without
any fine-tuning at all. If Phase 4 and Phase 10 use different prompts, the
difference between their scores measures prompt engineering, not fine-tuning.

`PROMPT_VERSION` is recorded in every result file. Change the template, bump
the version, and re-run the baseline — never compare across versions.
"""

from __future__ import annotations

import hashlib

PROMPT_VERSION = "v1"

# Kept deliberately plain. This is a *baseline*: it should represent what a
# competent engineer writes on the first attempt, not a prompt tuned against
# the test set. Tuning it against test data would leak the test set into the
# baseline and understate whatever fine-tuning later contributes.
SYSTEM_PROMPT = """You are an expert PostgreSQL analyst. You convert business \
questions into correct, executable PostgreSQL queries.

Rules:
- Output ONLY the SQL query. No explanation, no commentary.
- Use only the tables and columns given in the schema.
- Write a single SELECT statement. Never modify data.
- Use PostgreSQL syntax.
- When a foreign key is nullable, consider whether LEFT JOIN is needed to \
avoid silently dropping rows."""

USER_TEMPLATE = """Database schema:

{schema}

Question: {question}

PostgreSQL query:"""


def build_messages(question: str, schema: str) -> list[dict[str, str]]:
    """Return chat messages for the model.

    ``/no_think`` disables Qwen3's reasoning mode. Three reasons this matters
    for a benchmark:

    * Phase 9 fine-tunes the model to emit SQL directly. Benchmarking the base
      model *with* reasoning and the fine-tuned model *without* it would
      compare two different inference modes, not two sets of weights.
    * Reasoning traces cost hundreds of extra tokens per example, inflating
      latency and obscuring the generation cost being measured.
    * Reasoning output is far more sensitive to sampling, which undermines
      reproducibility.

    The token is harmless on models that do not recognise it.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(schema=schema, question=question)
                       + " /no_think",
        },
    ]


def prompt_fingerprint() -> str:
    """Hash of the template text, recorded alongside results."""
    combined = SYSTEM_PROMPT + "\x00" + USER_TEMPLATE + "\x00" + PROMPT_VERSION
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
