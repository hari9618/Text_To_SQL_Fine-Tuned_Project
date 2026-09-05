"""The prompt template used to ask a model to fix failing SQL.

Phase 11.

Deliberately kept free of imports so it can be shipped verbatim to a GPU host,
exactly like `src/model/prompt.py`. Versioned and hashed for the same reason:
a reworded repair prompt changes the repair success rate without touching the
weights, and Phase 14's ablation must be able to tell those apart.
"""

from __future__ import annotations

import hashlib

REPAIR_PROMPT_VERSION = "v1"

# Plain on purpose. This is the first repair prompt, not one tuned against the
# failures it is measured on -- tuning it on those 19 test failures would leak
# the test set into the repair result.
REPAIR_SYSTEM_PROMPT = """You are an expert PostgreSQL analyst. A query you \
wrote failed. You will be given the original question, the database schema, \
the failing query and the exact error.

Rules:
- Output ONLY the corrected SQL query. No explanation, no commentary.
- Fix the specific error reported. Do not rewrite what already worked.
- Use only the tables and columns given in the schema.
- Write a single SELECT statement. Never modify data.
- Use PostgreSQL syntax. Functions from other dialects do not exist here."""

REPAIR_TEMPLATE = """Database schema:

{schema}

Question: {question}

Failing query:
{failed_sql}

Error: {error}

Corrected PostgreSQL query:"""


def build_repair_messages(
    question: str, schema: str, failed_sql: str, error: str
) -> list[dict[str, str]]:
    """Chat messages asking the model to fix one failing query.

    ``/no_think`` matches the generation prompt so both calls run in the same
    inference mode; mixing reasoning and non-reasoning modes inside one
    pipeline would make the latency and cost figures incomparable.
    """
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": REPAIR_TEMPLATE.format(
                schema=schema, question=question,
                failed_sql=failed_sql or "(no statement produced)",
                error=error,
            ) + " /no_think",
        },
    ]


def repair_prompt_fingerprint() -> str:
    """Hash of the repair template, recorded alongside results."""
    combined = (REPAIR_SYSTEM_PROMPT + "\x00" + REPAIR_TEMPLATE + "\x00"
                + REPAIR_PROMPT_VERSION)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
