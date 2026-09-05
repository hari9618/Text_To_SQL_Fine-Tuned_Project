"""Apply database/schema.sql to the project database, then report what exists.

Phase 1, Task 1.2.

Runs as the application role, not the superuser — if the DDL needs rights the
app role does not have, that is a problem worth discovering now rather than in
production.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/load_schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sql.config import PROJECT_ROOT, ConfigError, app_config  # noqa: E402

SCHEMA_FILE = PROJECT_ROOT / "database" / "schema.sql"


def summarise(conn: psycopg.Connection) -> None:
    """Print table row counts and constraint totals for verification."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.relname,
                   (SELECT COUNT(*) FROM information_schema.columns
                     WHERE table_schema = 'public' AND table_name = c.relname)
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public' AND c.relkind = 'r'
            ORDER BY c.relname
            """
        )
        tables = cur.fetchall()

        cur.execute(
            """
            SELECT contype, COUNT(*)
            FROM pg_constraint con
            JOIN pg_namespace n ON n.oid = con.connamespace
            WHERE n.nspname = 'public'
            GROUP BY contype
            ORDER BY contype
            """
        )
        constraints = dict(cur.fetchall())

        cur.execute(
            """
            SELECT COUNT(*) FROM pg_indexes
            WHERE schemaname = 'public' AND indexname LIKE 'idx_%'
            """
        )
        index_count = cur.fetchone()[0]

    print(f"\nTables created: {len(tables)}")
    for name, columns in tables:
        print(f"  {name:<15} {columns:>3} columns")

    labels = {
        "p": "PRIMARY KEY",
        "f": "FOREIGN KEY",
        "u": "UNIQUE",
        "c": "CHECK",
    }
    print("\nConstraints:")
    for code, label in labels.items():
        print(f"  {label:<12} {constraints.get(code, 0)}")
    print(f"\nExplicit indexes (idx_*): {index_count}")


def main() -> int:
    if not SCHEMA_FILE.exists():
        print(f"[error] schema file not found: {SCHEMA_FILE}", file=sys.stderr)
        return 2

    try:
        cfg = app_config()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    ddl = SCHEMA_FILE.read_text(encoding="utf-8")
    print(f"Applying {SCHEMA_FILE.relative_to(PROJECT_ROOT)} to {cfg.database!r} "
          f"as {cfg.user!r}...")

    try:
        with psycopg.connect(cfg.conninfo(), connect_timeout=10) as conn:
            # schema.sql wraps itself in BEGIN/COMMIT, so a syntax error part
            # way through rolls the whole thing back — never a half-built schema.
            with conn.cursor() as cur:
                cur.execute(ddl)
            summarise(conn)
    except psycopg.Error as exc:
        print(f"\n[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\n[ok] schema applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
