# `dataset/`

The Text-to-SQL benchmark: natural-language questions paired with the SQL that
answers them, against the `enterprise_sql` database.

This is the measuring instrument for the whole project. Every later claim —
that schema retrieval helps, that fine-tuning helps, that SQL repair helps — is
a number produced here. If the benchmark is wrong, every one of those numbers is
wrong in a way no amount of downstream work can fix.

---

## Layout

```
dataset/
├── generation/    template library and value pools (Phase 2)
├── raw/           source material, if any is ever imported
├── generated/     benchmark.jsonl + statistics.json  (Phase 2 output)
├── validated/     examples that survive execution     (Phase 3)
├── train/         fine-tuning split                   (Phase 3)
├── validation/    hyperparameter split                (Phase 3)
└── test/          held-out evaluation split           (Phase 3)
```

Generation and validation are deliberately separate directories and separate
scripts. A generator that validated its own output would be marking its own
homework — Phase 3 checks the SQL by *executing* it, with no knowledge of how it
was produced.

---

## How generation works

Questions come from **templates**, not free-form invention:

```python
T("e01", "easy", "sales", "select where",
  ["Show all customers from {country}.",
   "List the customers located in {country}.",
   "Which customers are based in {country}?"],
  "SELECT customer_id, customer_name, country FROM customers "
  "WHERE country = '{country}'")
```

Each template carries several phrasings and one SQL query. Slots written
`{like_this}` are filled with the same value in both.

**Slot values come from the live database**, never invented. `{country}` is
filled from `SELECT DISTINCT country FROM customers`. Two reasons this matters:

- The SQL returns rows. A question whose correct answer is an empty result set
  cannot distinguish a correct query from a subtly wrong one — both return
  nothing.
- Inventing `'Wakanda'` would teach the model that plausible-sounding literals
  are acceptable, which is exactly the hallucination behaviour this project
  exists to measure and reduce.

**Multiple phrasings per template** are the point, not padding. A model trained
on one way of asking "how many customers are in India" learns that sentence, not
the task. Three phrasings of the same query force it to generalise.

---

## Correctness rules enforced at generation time

Three validators run before anything is written. Each guards against a defect
that produces data which *looks* fine and silently corrupts scoring.

**Unique template ids.** Ids drive the train/test split; a duplicate would put
the same pattern on both sides.

**Every phrasing mentions every slot the SQL uses.** If the SQL filters on
`{days}` but a phrasing does not mention it, all values of `{days}` collapse to
one question text with different SQL behind it — one question, several
conflicting gold answers.

**No two templates share a phrasing unless their SQL matches.** This caught a
real collision: `m35` and `h28` both rendered "Which departments have an average
salary above 100000?", one returning `department_id` and the other
`department_name`. Both defensible; together, unscoreable.

---

## Splitting: by template, never by row

The train/validation/test split happens in Phase 3, and it splits on
`template_id`, not on individual examples.

Random row-level splitting would be **test leakage**. All three phrasings of
`e01` with `{country} = 'India'` are near-identical; scattering them across
train and test would let the model memorise the pattern in training and score
highly on test without generalising at all. The reported accuracy would be
inflated and meaningless.

Splitting by template means test questions use phrasings and value combinations
the model has never seen for query patterns it has never seen.

---

## Current contents

Produced by `scripts/generate_benchmark.py` with `RANDOM_SEED = 20260808`,
then validated by `scripts/validate_dataset.py`:

| | |
|---|---|
| Templates | 174 |
| Generated examples | 3,045 |
| **Validated examples** | **3,045 (100 %)** |
| Unique SQL queries | 1,015 |
| Execution failures | 0 |
| Examples using `NOW()` | 0 |

**By difficulty**

| Tier | Examples | SQL features |
|---|---:|---|
| hard | 963 | JOIN, multi-JOIN, subquery, CTE |
| easy | 852 | SELECT, WHERE, ORDER BY, LIMIT |
| enterprise | 567 | ambiguity, NULL handling, business rules, cross-domain |
| medium | 426 | GROUP BY, HAVING, COUNT, SUM, AVG |
| very_hard | 237 | window functions, nested aggregation, date arithmetic |

**By domain**

| Domain | Examples |
|---|---:|
| sales | 1,326 |
| catalogue | 672 |
| cross_domain | 375 |
| logistics | 249 |
| hr | 243 |
| finance | 180 |

**Splits** — train 2,133 · validation 459 · test 453. Leakage-free by
construction; see [`validated/VALIDATION_REPORT.md`](validated/VALIDATION_REPORT.md).

## Time-relative questions are anchored, not clock-dependent

Questions like "orders pending more than 90 days" resolve against a fixed
`DATA_AS_OF` (`src/constants.py`), never `NOW()`. Templates write `{as_of}` and
the constructor bakes in the literal timestamp before slot filling.

Without this, a gold answer moves every day — so the Phase 5 baseline and the
Phase 10 fine-tuned run would be graded against *different correct answers*
while still producing plausible numbers. Nine days of drift already changed
three of the four affected gold result sets.

## Thresholds come from the data, not from guesses

Numeric slots are **interior deciles of the column each threshold is compared
against**, computed at generation time:

```
{employee_salary}   deciles of employees.salary
{dept_avg_salary}   deciles of AVG(salary) per department
{product_price}     deciles of products.unit_price
{category_avg_price} deciles of AVG(unit_price) per category
```

A slot is named for its column, so two templates comparing against different
distributions never share a pool. Bounds are excluded, so every threshold splits
the rows into two non-empty parts — the predicate always does real work.

This replaced hand-written lists that caused two problems: predicates matching
nothing (credit limit "below 1,000" when the minimum is 5,000) and predicates
matching everything (`salary > 40000` and `> 50000` returning identical rows,
so a model ignoring the threshold still scored correct).

---

## On dataset size

3,051 examples from 1,017 distinct SQL queries. No arbitrary size target is
being chased.

The ceiling is structural: **102 of the 174 templates have no slots.** "What is
the average order value?" has one correct query and no parameter to vary, so it
contributes three examples — one per phrasing — and cannot honestly contribute
more. The rest are bounded by real cardinality: `{country}` has 12 values
because the database contains 12 countries.

Total examples are therefore roughly `unique SQL × phrasings`. Growing it
honestly means more templates or more phrasings, not more rows per query.

What is deliberately *not* done: duplicating rows, loosening the grounding rule,
or inventing literals to inflate the count. A benchmark padded with
near-duplicates reports optimistic accuracy that does not survive contact with
real questions.

For scale: Spider, the standard academic Text-to-SQL benchmark, has 1,034
development examples and 2,147 test examples.

---

## Reproducibility

Generation is deterministic. Same seed, same database, same 2,949 examples in
the same order. Combined with the reproducible database (`generate_data.py`,
same seed), the entire evaluation setup can be rebuilt exactly — which is what
makes the base-model and fine-tuned numbers comparable at all.
