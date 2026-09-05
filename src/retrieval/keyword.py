"""Lexical schema retriever with foreign-key path expansion.

Version 1 of retrieval: no embeddings, no extra dependencies. That is a
deliberate starting point — a lexical baseline is cheap, fully interpretable,
and establishes whether an embedding model would actually earn its 8 GB-RAM
cost. Measure first, then decide.

Two stages:

**Scoring** ranks tables by how well the question matches their identifiers,
their column documentation, and — most importantly — their *values*.

**Path expansion** then repairs the fatal weakness of scoring alone. "Which
customers bought products from more than 3 categories?" mentions customers,
products and categories, and no lexical signal whatsoever points at ``orders``
or ``order_items``. Yet without those two bridge tables the join is impossible,
and the model can only respond by inventing a relationship. Retrieval that
drops a required table does not merely lose accuracy, it *causes* schema
hallucination — the exact failure it was introduced to reduce.

So selected tables are connected along shortest paths through the foreign-key
graph, and every table on those paths is included.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from src.model.schema_context import ForeignKey, Table
from src.retrieval.schema_index import SchemaIndex, normalise, tokenise

# Relative weights. Values score highest because a matching literal is nearly
# unambiguous evidence ('delivered' appears only in orders.status), whereas an
# identifier match can be coincidental ("order" appears in several tables).
WEIGHT_TABLE_NAME = 3.0
WEIGHT_COLUMN_NAME = 1.0
WEIGHT_COMMENT = 0.4
WEIGHT_VALUE = 2.5

DEFAULT_MAX_TABLES = 6
# Tables scoring below this fraction of the best score are dropped as noise.
RELATIVE_SCORE_FLOOR = 0.15


@dataclass
class RetrievalResult:
    """What the retriever chose, and why."""

    tables: list[str]
    seed_tables: list[str]
    bridge_tables: list[str]
    scores: dict[str, float]
    matched_values: dict[str, list[str]] = field(default_factory=dict)
    schema_text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "tables": self.tables,
            "seed_tables": self.seed_tables,
            "bridge_tables": self.bridge_tables,
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
            "matched_values": self.matched_values,
            "schema_chars": len(self.schema_text),
        }


def _shortest_path(index: SchemaIndex, start: str, goal: str) -> list[str]:
    """Breadth-first path through the foreign-key graph, inclusive of ends."""
    if start == goal:
        return [start]
    seen = {start}
    queue: deque[list[str]] = deque([[start]])
    while queue:
        path = queue.popleft()
        for neighbour in index.neighbours(path[-1]):
            if neighbour in seen:
                continue
            if neighbour == goal:
                return path + [neighbour]
            seen.add(neighbour)
            queue.append(path + [neighbour])
    return []


def score_tables(
    index: SchemaIndex, question: str
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Score every table against the question."""
    tokens = tokenise(question)
    terms = normalise(tokens)
    lowered = question.lower()

    scores: dict[str, float] = {t.name: 0.0 for t in index.tables}
    matched: dict[str, list[str]] = {}

    for name, ti in index.index.items():
        table_terms = normalise(tokenise(name))
        if table_terms & terms:
            scores[name] += WEIGHT_TABLE_NAME * len(table_terms & terms)

        for column, column_terms in ti.column_terms.items():
            overlap = column_terms & terms
            if overlap:
                # A full column-name match counts more than one shared token:
                # "order_date" matching both words beats matching only "date".
                scores[name] += WEIGHT_COLUMN_NAME * (
                    len(overlap) / max(len(column_terms), 1)
                ) * len(overlap)

        overlap = ti.comment_terms & terms
        if overlap:
            scores[name] += WEIGHT_COMMENT * len(overlap)

    # Value matching. Multi-word literals are checked as substrings of the raw
    # question so "United States" and "credit card" match despite tokenising
    # into pieces that individually mean little.
    for value, locations in index.value_index.items():
        hit = (value in lowered) if " " in value else (value in terms or value in tokens)
        if not hit:
            continue
        # A literal appearing in many columns is weak evidence; one appearing
        # in a single column is nearly decisive.
        weight = WEIGHT_VALUE / len(locations)
        for table, column in locations:
            scores[table] += weight
            matched.setdefault(table, [])
            label = f"{column}='{value}'"
            if label not in matched[table] and len(matched[table]) < 6:
                matched[table].append(label)

    return scores, matched


def select_tables(
    index: SchemaIndex,
    scores: dict[str, float],
    max_tables: int = DEFAULT_MAX_TABLES,
    expand_neighbours: int = 0,
) -> tuple[list[str], list[str]]:
    """Pick seed tables by score, then add the bridges needed to join them."""
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best = ranked[0][1] if ranked else 0.0

    if best <= 0:
        # Nothing matched. Falling back to the full schema is the safe failure:
        # a wrong subset guarantees a wrong answer, whereas the full schema is
        # merely the Phase 5 baseline condition.
        return [t.name for t in index.tables], []

    seeds = [
        name for name, score in ranked[:max_tables]
        if score >= best * RELATIVE_SCORE_FLOOR and score > 0
    ]

    selected = set(seeds)
    bridges: set[str] = set()
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            path = _shortest_path(index, a, b)
            for node in path[1:-1]:
                if node not in selected:
                    bridges.add(node)
                    selected.add(node)

    # Detail tables are invisible to lexical matching. "Highest spending
    # customers" needs order_items, but the word "spending" appears nowhere in
    # that table's identifiers, comments or values -- and because customers and
    # orders are directly adjacent, path expansion adds no bridge either.
    #
    # Pulling in the immediate neighbours of the best-scoring seeds catches
    # them. The trade is deliberate and asymmetric: a missing table makes a
    # correct answer impossible and provokes hallucination, whereas an extra
    # table costs only prompt tokens.
    if expand_neighbours > 0:
        for seed in seeds[:expand_neighbours]:
            for neighbour in index.neighbours(seed):
                if neighbour not in selected:
                    bridges.add(neighbour)
                    selected.add(neighbour)

    order = {t.name: i for i, t in enumerate(index.tables)}
    return sorted(selected, key=lambda n: order.get(n, 999)), sorted(bridges)


def render_subset(index: SchemaIndex, table_names: list[str]) -> str:
    """Render only the selected tables, in the same DDL style as the baseline.

    Identical formatting to the full-schema renderer is essential: if retrieval
    also changed how the schema is written, the measured difference would mix
    retrieval with prompt formatting and neither could be attributed.
    """
    chosen = set(table_names)
    blocks: list[str] = []

    for name in table_names:
        table: Table | None = index.by_name.get(name)
        if table is None:
            continue
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
    for name in table_names:
        table = index.by_name.get(name)
        if table is None:
            continue
        for fk in table.foreign_keys:
            # Only relationships whose other end is also present; a dangling
            # reference would invite a join against a table the model cannot see.
            if fk.references_table not in chosen:
                continue
            note = ("  [nullable: use LEFT JOIN to keep unmatched rows]"
                    if fk.nullable else "")
            relationships.append(
                f"--   {table.name}.{fk.column} -> "
                f"{fk.references_table}.{fk.references_column}{note}"
            )

    return "\n\n".join(blocks) + "\n\n" + "\n".join(relationships)


class KeywordRetriever:
    """Retrieve a relevant schema subset for a question."""

    name = "keyword_v1"

    def __init__(self, index: SchemaIndex, max_tables: int = DEFAULT_MAX_TABLES,
                 expand_neighbours: int = 0):
        self.index = index
        self.max_tables = max_tables
        self.expand_neighbours = expand_neighbours

    def retrieve(self, question: str) -> RetrievalResult:
        scores, matched = score_tables(self.index, question)
        tables, bridges = select_tables(
            self.index, scores, self.max_tables, self.expand_neighbours)
        seeds = [t for t in tables if t not in bridges]
        return RetrievalResult(
            tables=tables,
            seed_tables=seeds,
            bridge_tables=bridges,
            scores={k: v for k, v in scores.items() if v > 0},
            matched_values={k: v for k, v in matched.items() if k in tables},
            schema_text=render_subset(self.index, tables),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "retriever": self.name,
            "method": "lexical identifier + comment + value matching, "
                      "foreign-key shortest-path expansion",
            "max_seed_tables": self.max_tables,
            "expand_neighbours_of_top_n_seeds": self.expand_neighbours,
            "relative_score_floor": RELATIVE_SCORE_FLOOR,
            "weights": {
                "table_name": WEIGHT_TABLE_NAME,
                "column_name": WEIGHT_COLUMN_NAME,
                "comment": WEIGHT_COMMENT,
                "value": WEIGHT_VALUE,
            },
            "indexed_value_columns": len(self.index.indexed_columns),
            "embeddings": False,
        }
