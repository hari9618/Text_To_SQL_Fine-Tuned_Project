"""Database connection settings, loaded from the environment.

Credentials are never hard-coded and never committed. Values come from a
git-ignored ``.env`` at the project root (see ``.env.example``), or from real
environment variables, which take precedence — that is what lets the same code
run unchanged in Docker or CI later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root is two levels up from this file: src/sql/config.py -> project/
# Resolved relative to __file__ so nothing depends on the current directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# override=False: a real environment variable beats the .env file.
load_dotenv(PROJECT_ROOT / ".env", override=False)


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _require(name: str) -> str:
    """Read an environment variable, failing loudly if it is absent.

    Silent fallbacks to defaults hide misconfiguration until something fails
    much later with a confusing error, so missing values raise immediately.
    """
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"Required environment variable {name!r} is not set. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


@dataclass(frozen=True)
class DatabaseConfig:
    """Connection settings for a single PostgreSQL role."""

    host: str
    port: int
    database: str
    user: str
    password: str

    def conninfo(self) -> str:
        """Return a libpq connection string for psycopg."""
        return (
            f"host={self.host} port={self.port} dbname={self.database} "
            f"user={self.user} password={self.password}"
        )

    def __repr__(self) -> str:
        """Redact the password so it cannot leak into logs or tracebacks."""
        return (
            f"DatabaseConfig(host={self.host!r}, port={self.port}, "
            f"database={self.database!r}, user={self.user!r}, password='***')"
        )


def _port() -> int:
    raw = os.getenv("PGPORT", "5432").strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"PGPORT must be an integer, got {raw!r}") from exc


def target_database_name() -> str:
    """Name of the project database, e.g. ``enterprise_sql``."""
    return os.getenv("PGDATABASE", "enterprise_sql").strip()


def superuser_config(database: str | None = None) -> DatabaseConfig:
    """Superuser connection, used only for setup tasks.

    Creating a database or role requires elevated rights. Everything after
    setup uses :func:`app_config` instead.

    Args:
        database: Which database to connect to. Defaults to ``postgres``, the
            maintenance database, because you cannot create a database while
            connected to the one being created.
    """
    return DatabaseConfig(
        host=os.getenv("PGHOST", "localhost").strip(),
        port=_port(),
        database=database or "postgres",
        user=os.getenv("PGSUPERUSER", "postgres").strip(),
        password=_require("PGSUPERPASSWORD"),
    )


def app_config() -> DatabaseConfig:
    """Application connection used by the Text-to-SQL pipeline.

    Deliberately not the superuser: this role executes model-generated SQL, and
    a language model must never hold rights to drop tables.
    """
    return DatabaseConfig(
        host=os.getenv("PGHOST", "localhost").strip(),
        port=_port(),
        database=target_database_name(),
        user=_require("APP_DB_USER"),
        password=_require("APP_DB_PASSWORD"),
    )
