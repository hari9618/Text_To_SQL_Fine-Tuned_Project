"""A searchable index over the database schema.

Three signals are indexed, because a question rarely names the table it needs.

*Identifiers* — table and column names, tokenised and crudely singularised, so
"customers" in a question matches the ``customers`` table and "order date"
matches ``orders.order_date``.

*Documentation* — the ``COMMENT ON`` text written in Phase 2. That is where a
column's business meaning lives ("NULL for self-service online orders"), and it
is often the only place a question's vocabulary appears at all.

*Values* — the distinct contents of low-cardinality text columns. This is the
strongest available signal and the one a naive retriever misses entirely: the
question "How many customers are from India?" contains no schema identifier
whatsoever. Only the literal 'India' points at ``customers.country``.

A value index is affordable only because the columns worth indexing are
low-cardinality by definition. Indexing ``customer_name`` would be pointless —
12,000 unique values no question repeats verbatim — and is skipped.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import psycopg

from src.model.schema_context import Table, load_schema

# Above this many distinct values a column is treated as free text rather than
# a controlled vocabulary, and its contents are not indexed.
MAX_DISTINCT_VALUES_TO_INDEX = 400

# Text-like column types worth indexing values for.
INDEXABLE_TYPES = ("VARCHAR", "CHAR", "TEXT")

# Words carrying no retrieval signal. Deliberately short: an aggressive
# stopword list strips domain terms that matter here ("order", "count").
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in",
    "on", "at", "to", "for", "from", "by", "with", "and", "or", "as", "that",
    "this", "these", "those", "it", "its", "we", "our", "us", "you", "your",
    "show", "list", "give", "display", "find", "get", "which", "what", "who",
    "whom", "whose", "where", "when", "how", "many", "much", "all", "any",
    "each", "every", "do", "does", "did", "have", "has", "had", "please",
    "me", "my", "there", "their", "them", "than", "then", "into", "out",
}

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords."""
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in STOPWORDS]


def singularise(token: str) -> str:
    """Crude plural stripping so 'customers' and 'customer' match.

    A real stemmer would be overkill: schema identifiers are already normalised
    snake_case English nouns, and over-stemming (matching the wrong table) is a
    worse failure than missing a rare plural.
    """
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalise(tokens: list[str]) -> set[str]:
    return {singularise(t) for t in tokens}


@dataclass
class TableIndex:
    """Everything searchable about one table."""

    name: str
    identifier_terms: set[str] = field(default_factory=set)
    comment_terms: set[str] = field(default_factory=set)
    column_terms: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class SchemaIndex:
    tables: list[Table]
    by_name: dict[str, Table] = field(default_factory=dict)
    index: dict[str, TableIndex] = field(default_factory=dict)
    # normalised value or value-token -> {(table, column)}
    value_index: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    # undirected foreign-key adjacency: table -> {neighbour: (col, ref_col)}
    graph: dict[str, dict[str, tuple[str, str]]] = field(default_factory=dict)
    indexed_columns: list[tuple[str, str, int]] = field(default_factory=list)

    def neighbours(self, table: str) -> set[str]:
        return set(self.graph.get(table, {}))


def _build_graph(tables: list[Table]) -> dict[str, dict[str, tuple[str, str]]]:
    """Undirected join graph. Direction is irrelevant for reachability."""
    graph: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    for table in tables:
        graph.setdefault(table.name, {})
        for fk in table.foreign_keys:
            if fk.references_table == table.name:
                continue  # a self-reference adds no new table to a join path
            graph[table.name][fk.references_table] = (fk.column, fk.references_column)
            graph[fk.references_table][table.name] = (fk.references_column, fk.column)
    return dict(graph)


def _index_values(
    conn: psycopg.Connection, tables: list[Table]
) -> tuple[dict[str, set[tuple[str, str]]], list[tuple[str, str, int]]]:
    values: dict[str, set[tuple[str, str]]] = defaultdict(set)
    indexed: list[tuple[str, str, int]] = []

    with conn.cursor() as cur:
        for table in tables:
            for column in table.columns:
                if not any(t in column.data_type for t in INDEXABLE_TYPES):
                    continue
                cur.execute(
                    'SELECT COUNT(DISTINCT "{}") FROM "{}"'.format(
                        column.name, table.name)
                )
                distinct = cur.fetchone()[0] or 0
                if distinct == 0 or distinct > MAX_DISTINCT_VALUES_TO_INDEX:
                    continue
                cur.execute(
                    'SELECT DISTINCT "{}" FROM "{}" WHERE "{}" IS NOT NULL'.format(
                        column.name, table.name, column.name)
                )
                for (value,) in cur.fetchall():
                    text = str(value).strip().lower()
                    if len(text) < 2:
                        continue
                    values[text].add((table.name, column.name))
                    # Index single tokens too, so "credit card" matches
                    # 'credit_card' and "United States" matches either word.
                    for token in tokenise(text):
                        if len(token) > 2:
                            values[token].add((table.name, column.name))
                indexed.append((table.name, column.name, distinct))
    return dict(values), indexed


def build_index(conn: psycopg.Connection) -> SchemaIndex:
    tables = load_schema(conn)
    idx = SchemaIndex(tables=tables, by_name={t.name: t for t in tables})
    idx.graph = _build_graph(tables)

    for table in tables:
        ti = TableIndex(name=table.name)
        ti.identifier_terms = normalise(tokenise(table.name))
        for column in table.columns:
            terms = normalise(tokenise(column.name))
            ti.column_terms[column.name] = terms
            ti.identifier_terms |= terms
            if column.comment:
                ti.comment_terms |= normalise(tokenise(column.comment))
        idx.index[table.name] = ti

    idx.value_index, idx.indexed_columns = _index_values(conn, tables)
    return idx
