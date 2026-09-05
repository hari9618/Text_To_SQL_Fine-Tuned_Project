"""Project-wide constants that must stay identical across every phase.

The values here are not configuration — they are not meant to vary per
environment, and changing one invalidates work already produced. They live in
``src/`` so that dataset generation, validation, evaluation, and the API all
import the same definition rather than each hard-coding its own copy.
"""

from __future__ import annotations

from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# DATA_AS_OF — the instant the dataset is notionally "current".
#
# Every time-relative question is anchored here instead of to the wall clock.
# "Orders pending for more than 90 days" means 90 days before this timestamp,
# not 90 days before whenever the query happens to run.
#
# Why this matters more than it looks:
#
# The project's central claim is measured by comparing a base model (Phase 5)
# against a fine-tuned model (Phase 10) on the same benchmark. Those runs are
# weeks apart. With NOW(), a gold query's result set drifts between them, so
# the two models would be graded against different correct answers and the
# comparison would be meaningless — while still producing plausible numbers.
#
# The value sits one month after the last generated order (2026-07-01), leaving
# room for the longest downstream chain (ship up to 7 days after the order,
# deliver up to 14 days later) to complete without any row being dated in the
# future.
#
# Changing this constant invalidates every recorded result fingerprint. Treat
# it as fixed for the life of the project.
# ---------------------------------------------------------------------------

DATA_AS_OF: datetime = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)

# The same instant as a PostgreSQL literal, derived from the value above so the
# two can never drift apart. Embedded directly into gold SQL, which is what
# makes generation, validation, and evaluation agree by construction rather
# than by convention.
#
#   TIMESTAMPTZ '2026-08-01 00:00:00+00:00'
DATA_AS_OF_SQL: str = f"TIMESTAMPTZ '{DATA_AS_OF.isoformat(sep=' ')}'"
