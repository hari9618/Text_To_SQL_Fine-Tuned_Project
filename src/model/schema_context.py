"""Render the database schema as text for the model.

Phase 4.

The schema is read from the **live PostgreSQL catalog**, not from
`database/schema.sql`. Those two should agree, but if they ever diverge the
catalog is what queries actually run against — showing the model a stale file
would create hallucinations that are really documentation bugs.

Phase 4 shows the *entire* schema. That is the point: it is ablation
configuration 1, "base model with no retrieval". Phase 6 will show a retrieved
subset instead, and the difference between the two numbers is the measured
value of retrieval. Trimming the schema here would quietly borrow Phase 6's
benefit and make retrieval look worthless later.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import psycopg


@dataclass
class Column:
    name: str
    data_type: str
    nullable: bool
    comment: str | None = None


@dataclass
class ForeignKey:
    column: str
    references_table: str
    references_column: str
    nullable: bool


@dataclass
class Table:
    name: str
    columns: list[Column] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)


def _short_type(data_type: str, char_len: int | None,
                precision: int | None, scale: int | None) -> str:
    """Compact type names. `VARCHAR(200)` reads better than the catalog's
    `character varying`, and shorter types mean fewer prompt tokens."""
    mapping = {
        "character varying": f"VARCHAR({char_len})" if char_len else "VARCHAR",
        "character": f"CHAR({char_len})" if char_len else "CHAR",
        "integer": "INTEGER",
        "bigint": "BIGINT",
        "smallint": "SMALLINT",
        "boolean": "BOOLEAN",
        "text": "TEXT",
        "date": "DATE",
        "timestamp with time zone": "TIMESTAMPTZ",
        "timestamp without time zone": "TIMESTAMP",
        "numeric": (
            f"NUMERIC({precision},{scale})" if precision is not None else "NUMERIC"
        ),
    }
    return mapping.get(data_type, data_type.upper())


def load_schema(conn: psycopg.Connection) -> list[Table]:
    """Read tables, columns, keys and comments from the catalog."""
    tables: dict[str, Table] = {}

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.table_name, c.column_name, c.data_type,
                   c.character_maximum_length, c.numeric_precision,
                   c.numeric_scale, c.is_nullable,
                   col_description(pgc.oid, c.ordinal_position) AS comment
            FROM information_schema.columns c
            JOIN pg_class pgc ON pgc.relname = c.table_name
            JOIN pg_namespace n ON n.oid = pgc.relnamespace AND n.nspname = 'public'
            WHERE c.table_schema = 'public'
            ORDER BY c.table_name, c.ordinal_position
            """
        )
        for (table, column, dtype, char_len, prec, scale, nullable,
             comment) in cur.fetchall():
            tables.setdefault(table, Table(name=table)).columns.append(
                Column(
                    name=column,
                    data_type=_short_type(dtype, char_len, prec, scale),
                    nullable=(nullable == "YES"),
                    comment=comment,
                )
            )

        cur.execute(
            """
            SELECT tc.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON kcu.constraint_name = tc.constraint_name
             AND kcu.table_schema = tc.table_schema
            WHERE tc.table_schema = 'public'
              AND tc.constraint_type = 'PRIMARY KEY'
            ORDER BY tc.table_name, kcu.ordinal_position
            """
        )
        for table, column in cur.fetchall():
            if table in tables:
                tables[table].primary_key.append(column)

        # Nullability travels with the foreign key because it decides whether a
        # correct query needs LEFT JOIN or INNER JOIN — the single most common
        # source of wrong-but-runnable SQL in this schema.
        cur.execute(
            """
            SELECT src.relname, a.attname, tgt.relname, b.attname, a.attnotnull
            FROM pg_constraint con
            JOIN pg_class src ON src.oid = con.conrelid
            JOIN pg_class tgt ON tgt.oid = con.confrelid
            JOIN unnest(con.conkey)  WITH ORDINALITY AS ck(attnum, ord) ON TRUE
            JOIN unnest(con.confkey) WITH ORDINALITY AS fk(attnum, ord)
              ON fk.ord = ck.ord
            JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = ck.attnum
            JOIN pg_attribute b ON b.attrelid = con.confrelid AND b.attnum = fk.attnum
            WHERE con.contype = 'f'
            ORDER BY src.relname, a.attname
            """
        )
        for src, col, tgt, ref_col, notnull in cur.fetchall():
            if src in tables:
                tables[src].foreign_keys.append(
                    ForeignKey(column=col, references_table=tgt,
                               references_column=ref_col, nullable=not notnull)
                )

    return [tables[name] for name in sorted(tables)]


def render_schema(tables: list[Table]) -> str:
    """Format the schema as CREATE TABLE-like text.

    DDL-shaped rather than prose: the model saw enormous amounts of SQL DDL
    during pretraining, so this format is far more familiar to it than an
    invented table layout would be.
    """
    blocks: list[str] = []
    for table in tables:
        lines = [f"CREATE TABLE {table.name} ("]
        for column in table.columns:
            parts = [f"  {column.name} {column.data_type}"]
            if not column.nullable:
                parts.append("NOT NULL")
            if column.name in table.primary_key:
                parts.append("PRIMARY KEY")
            line = " ".join(parts)
            if column.comment:
                line += f"  -- {column.comment}"
            lines.append(line + ",")
        if lines[-1].endswith(","):
            lines[-1] = lines[-1][:-1]
        lines.append(");")
        blocks.append("\n".join(lines))

    relationships = ["-- Foreign key relationships (join paths):"]
    for table in tables:
        for fk in table.foreign_keys:
            note = "  [nullable: use LEFT JOIN to keep unmatched rows]" if fk.nullable else ""
            relationships.append(
                f"--   {table.name}.{fk.column} -> "
                f"{fk.references_table}.{fk.references_column}{note}"
            )

    return "\n\n".join(blocks) + "\n\n" + "\n".join(relationships)


def schema_fingerprint(schema_text: str) -> str:
    """Hash of the rendered schema, recorded for reproducibility.

    If this changes between the Phase 4 baseline and the Phase 10 evaluation,
    the two runs did not show the model the same database and their scores are
    not comparable.
    """
    return hashlib.sha256(schema_text.encode("utf-8")).hexdigest()[:16]


def build_schema_context(conn: psycopg.Connection) -> tuple[str, str]:
    """Return (rendered schema, fingerprint)."""
    text = render_schema(load_schema(conn))
    return text, schema_fingerprint(text)
