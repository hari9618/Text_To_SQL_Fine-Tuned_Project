"""Benchmark v2 and prompt v2 keep their promises.

Phase 15. These are structural guarantees, not accuracy claims:

* v1 is untouched: its prompt fingerprint and its templates are what they
  were when the published numbers were measured.
* v2's test split is the same 453 questions by id and the same 48 templates.
* Every v2 change is one of the five documented rules.
* The v2 prompt's reference date is the database's own ``DATA_AS_OF``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset.generation import templates as v1
from dataset.generation import templates_v2 as v2
from src import benchmark_versions
from src.constants import DATA_AS_OF_SQL
from src.model import prompt as prompt_v1
from src.model import prompt_v2

V1 = benchmark_versions.get("v1")
V2 = benchmark_versions.get("v2")


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------- prompts --

def test_v1_prompt_is_still_the_frozen_baseline_prompt():
    assert prompt_v1.prompt_fingerprint() == "8288e41a496531a9"


def test_v2_prompt_is_a_different_version_with_its_own_fingerprint():
    assert prompt_v2.PROMPT_VERSION == "v2"
    assert prompt_v2.prompt_fingerprint() != prompt_v1.prompt_fingerprint()


def test_v2_reference_date_is_the_databases_data_as_of():
    """The literal in the import-free prompt module must track the constant."""
    assert prompt_v2.REFERENCE_DATE_SQL == DATA_AS_OF_SQL
    assert DATA_AS_OF_SQL in prompt_v2.SYSTEM_PROMPT


def test_v2_prompt_keeps_v1_shape():
    msgs = prompt_v2.build_messages("How many customers?", "SCHEMA")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"].endswith(" /no_think")
    assert "SCHEMA" in msgs[1]["content"]
    # The glossary states the conventions the benchmark v2 gold follows.
    for phrase in ("SELECT *", "DATE_TRUNC('month'", "orders.total_amount",
                   "discount_pct", "'cancelled' or 'returned'", "NOW()"):
        assert phrase in prompt_v2.SYSTEM_PROMPT, phrase


# -------------------------------------------------------------- templates --

def test_v1_templates_unchanged_by_v2_module():
    """Importing v2 must not mutate v1 (dataclasses are frozen, but check)."""
    v1_by_id = {t.template_id: t for t in v1.ALL_TEMPLATES}
    untouched = set(v1_by_id) - v2.changed_template_ids()
    v2_by_id = {t.template_id: t for t in v2.ALL_TEMPLATES}
    assert all(v2_by_id[i] == v1_by_id[i] for i in untouched)
    assert len(untouched) == 131


def test_every_v2_change_is_a_documented_rule():
    reasons = {r for _, r in v2.CHANGELOG}
    assert all(r.startswith("rule ") for r in reasons)
    assert len(v2.CHANGELOG) == 48
    assert len(v2.ALL_TEMPLATES) == len(v1.ALL_TEMPLATES) + len(v2.NEW_TRAIN_TEMPLATES)


def test_whole_row_templates_select_star_and_keep_their_filters():
    v1_by_id = {t.template_id: t for t in v1.ALL_TEMPLATES}
    for t in v2.ALL_TEMPLATES:
        if t.template_id in v2.WHOLE_ROW_TEMPLATES:
            assert t.sql.upper().startswith("SELECT * FROM "), t.template_id
            # Everything after FROM is the v1 query's tail: the filter did not move.
            assert t.sql.split(" FROM ", 1)[1] == v1_by_id[t.template_id].sql.split(" FROM ", 1)[1]


def test_new_templates_cover_the_missing_constructs():
    sqls = [t.sql.upper() for t in v2.NEW_TRAIN_TEMPLATES]
    assert sum("PARTITION BY" in s for s in sqls) >= 3
    assert sum("EXTRACT(EPOCH" in s for s in sqls) >= 2
    v1_train = load(V1.split("train"))
    assert not any("PARTITION BY" in r["sql"].upper() for r in v1_train), \
        "premise of rule 5 no longer holds"


def test_v2_validators_pass():
    v2.validate_all()


# ----------------------------------------------------------------- splits --

@pytest.mark.skipif(not V2.split("test").exists(), reason="v2 dataset not built")
def test_v2_test_split_is_the_v1_test_split_by_id_and_template():
    t1, t2 = load(V1.split("test")), load(V2.split("test"))
    assert [r["id"] for r in t1] == [r["id"] for r in t2]
    assert {r["template_id"] for r in t1} == {r["template_id"] for r in t2}
    assert len(t2) == 453


@pytest.mark.skipif(not V2.split("test").exists(), reason="v2 dataset not built")
def test_v2_splits_share_no_template_and_new_templates_are_train_only():
    splits = {s: load(V2.split(s)) for s in ("train", "validation", "test")}
    tpl = {s: {r["template_id"] for r in rows} for s, rows in splits.items()}
    assert not (tpl["train"] & tpl["test"])
    assert not (tpl["validation"] & tpl["test"])
    assert not (tpl["train"] & tpl["validation"])
    assert v2.NEW_TEMPLATE_IDS <= tpl["train"]
    assert not (v2.NEW_TEMPLATE_IDS & (tpl["validation"] | tpl["test"]))


@pytest.mark.skipif(not V2.split("test").exists(), reason="v2 dataset not built")
def test_v2_test_questions_only_changed_where_a_paraphrase_was_reworded():
    t1 = {r["id"]: r for r in load(V1.split("test"))}
    changed = {r["template_id"] for r in load(V2.split("test"))
               if r["question"] != t1[r["id"]]["question"]}
    assert changed == set(v2.QUESTION_OVERRIDES)


def test_v1_test_file_is_byte_identical_to_the_published_one():
    """The one file this project promised never to modify."""
    import hashlib
    digest = hashlib.sha256(V1.split("test").read_bytes()).hexdigest()[:16]
    assert digest == "ec7ddcae4f9d90d4"
