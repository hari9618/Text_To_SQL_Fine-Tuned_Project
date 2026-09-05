"""Create the project database and application role, then verify the connection.

Phase 1, Task 1.1. Idempotent — safe to run repeatedly.

What it does, and why:

1. Connects as the PostgreSQL superuser to the ``postgres`` maintenance
   database. You cannot create a database while connected to it, so the
   connection has to be made somewhere else first.
2. Creates the application role ``texttosql_app`` if missing.
3. Creates the ``enterprise_sql`` database owned by that role.
4. Reconnects as the application role to prove the credentials work.

The application role is deliberately not a superuser. Later phases execute
model-generated SQL against this database, and a language model must never hold
rights to drop tables or read other databases.

Usage (from the project root):

    env\\Scripts\\python.exe scripts/init_db.py
"""

from __future__ import annotations

import sys

import psycopg
from psycopg import sql

# Allow `python scripts/init_db.py` from the project root without installing
# the package: put the project root on sys.path so `src.` imports resolve.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.sql.config import (  # noqa: E402
    ConfigError,
    DatabaseConfig,
    app_config,
    superuser_config,
    target_database_name,
)


def role_exists(conn: psycopg.Connection, role: str) -> bool:
    """Return True if a PostgreSQL role of this name already exists."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
        return cur.fetchone() is not None


def database_exists(conn: psycopg.Connection, database: str) -> bool:
    """Return True if a database of this name already exists."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
        return cur.fetchone() is not None


def create_role(conn: psycopg.Connection, role: str, password: str) -> None:
    """Create a login role. Identifiers use sql.Identifier, not f-strings.

    Role and database names cannot be passed as query parameters — PostgreSQL
    only parameterises *values*. psycopg's sql.Identifier quotes them safely,
    which is the correct way to interpolate an identifier.
    """
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(role), sql.Literal(password)
            )
        )


def create_database(conn: psycopg.Connection, database: str, owner: str) -> None:
    """Create a database owned by the given role."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE DATABASE {} OWNER {}").format(
                sql.Identifier(database), sql.Identifier(owner)
            )
        )


def verify_connection(cfg: DatabaseConfig) -> tuple[str, str, str]:
    """Connect and return (server_version, current_database, current_user)."""
    with psycopg.connect(cfg.conninfo(), connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version(), current_database(), current_user")
            row = cur.fetchone()
    assert row is not None
    return row


def main() -> int:
    try:
        super_cfg = superuser_config()  # connects to `postgres`
        app_cfg = app_config()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    db_name = target_database_name()
    print(f"Target database : {db_name}")
    print(f"Application role: {app_cfg.user}")
    print(f"Server          : {super_cfg.host}:{super_cfg.port}\n")

    # CREATE DATABASE cannot run inside a transaction block, so this
    # connection must be in autocommit mode.
    try:
        with psycopg.connect(
            super_cfg.conninfo(), autocommit=True, connect_timeout=10
        ) as conn:
            if role_exists(conn, app_cfg.user):
                print(f"[skip]    role {app_cfg.user!r} already exists")
            else:
                create_role(conn, app_cfg.user, app_cfg.password)
                print(f"[created] role {app_cfg.user!r}")

            if database_exists(conn, db_name):
                print(f"[skip]    database {db_name!r} already exists")
            else:
                create_database(conn, db_name, app_cfg.user)
                print(f"[created] database {db_name!r} owned by {app_cfg.user!r}")

    except psycopg.OperationalError as exc:
        print(f"\n[error] could not connect as superuser: {exc}", file=sys.stderr)
        print(
            "Check PGSUPERPASSWORD in .env, and that the "
            "postgresql-x64-18 service is running.",
            file=sys.stderr,
        )
        return 1

    # Prove the application credentials actually work against the new database.
    print("\nVerifying application connection...")
    try:
        version, database, user = verify_connection(app_cfg)
    except psycopg.OperationalError as exc:
        print(f"[error] application connection failed: {exc}", file=sys.stderr)
        return 1

    print(f"[ok] connected to {database!r} as {user!r}")
    print(f"[ok] {version.split(',')[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
