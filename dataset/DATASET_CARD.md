---
license: apache-2.0
task_categories:
  - table-question-answering
  - text-generation
language:
  - en
tags:
  - text-to-sql
  - sql
  - postgresql
  - semantic-parsing
  - benchmark
size_categories:
  - 1K<n<10K
pretty_name: Enterprise Text-to-SQL Benchmark (PostgreSQL, 12 tables)
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: validation
        path: validation.jsonl
      - split: test
        path: test.jsonl
---

# Enterprise Text-to-SQL Benchmark

**3,045 natural-language questions paired with executable PostgreSQL**, over a
12-table enterprise schema (sales, catalogue, logistics, HR).

Built to answer one question honestly: *does fine-tuning actually improve
text-to-SQL?* On this benchmark, a QLoRA fine-tune of Qwen3-8B took strict
execution accuracy from **10.82 % to 50.99 %** — and the benchmark is designed
so that number cannot be inflated by leakage or by string-matching.

| split | rows | templates |
|---|---:|---:|
| train | 2,133 | 74 |
| validation | 459 | 52 |
| **test** | **453** | **48** |

**All three splits share zero templates with each other** — 74 / 52 / 48, no
overlap in any pair. Splitting is by *template equivalence group*, never by row,
so the test set measures generalisation to query patterns the model has never
seen, not to new slot values inside familiar ones.

```python
from datasets import load_dataset

ds = load_dataset("<repo-id>")
print(ds["test"][0]["question"])   # "Show orders placed in 2023."
print(ds["test"][0]["sql"])        # SELECT order_id, customer_id, ...
```

---

## Why this exists

Most text-to-SQL benchmarks score by comparing SQL *strings*. That is wrong in
both directions: it marks correct queries wrong for cosmetic differences, and it
cannot tell you whether a query would actually run.

These two are different strings and identical in meaning, and any sane benchmark
must count both as correct:

```sql
SELECT COUNT(*)           FROM customers WHERE country = 'India';
SELECT COUNT(customer_id) FROM customers WHERE country = 'India';
```

So every example here ships an **execution fingerprint** — an MD5 over the
sorted, stringified rows the gold query returned against the reference database.
Correctness means *your query returns the same result set*, not that it looks
similar.

---

## Fields

| field | type | meaning |
|---|---|---|
| `id` | string | stable identifier, e.g. `bench-000322` |
| `question` | string | the natural-language question |
| `sql` | string | gold PostgreSQL query |
| `difficulty` | string | `easy` / `medium` / `hard` / `very_hard` / `enterprise` |
| `domain` | string | `sales`, `catalogue`, `logistics`, `hr` |
| `template_id` | string | generating template — **the unit splits are made on** |
| `phrasing_index` | int | which paraphrase of the template this is |
| `slot_values` | dict | the values substituted into the template |
| `referenced_tables` | list | tables the gold query touches |
| `execution` | dict | `row_count`, `columns`, `fingerprint`, `latency_ms` |
| `split` | string | `train` / `validation` / `test` |

The `execution.fingerprint` is what makes execution-based scoring possible
without shipping a database dump.

### Difficulty tiers

| tier | n (test) | what it exercises |
|---|---:|---|
| `easy` | 126 | SELECT, WHERE, ORDER BY, LIMIT |
| `medium` | 63 | GROUP BY, HAVING, COUNT/SUM/AVG |
| `hard` | 144 | joins, multiple joins, subqueries, CTEs |
| `very_hard` | 36 | window functions, nested aggregation, date arithmetic |
| `enterprise` | 84 | ambiguous terminology, NULL handling, business rules |

---

## The database

`schema.sql` rebuilds it; `schema_context.txt` is the rendered schema that was
put in the model's prompt (fingerprint `d03619e711661bc5`, 5,455 characters).

12 tables, ~328,000 rows, all synthetic — **no real personal data**. Generation
uses a fixed seed, so the database is reproducible, which is what makes the
execution fingerprints meaningful.

| table | rows | | table | rows |
|---|---:|---|---|---:|
| customers | 12,000 | | order_items | 154,069 |
| orders | 55,000 | | payments | 57,088 |
| products | 2,500 | | shipments | 40,850 |
| categories | 120 | | inventory | 6,266 |
| suppliers | 250 | | employees | 500 |
| warehouses | 15 | | departments | 12 |

Dates are anchored to a fixed `DATA_AS_OF` constant rather than `NOW()`, so
questions like *"orders in the last 90 days"* have a stable answer instead of
one that changes overnight.

---

## Results on this benchmark

Qwen3-8B, evaluated by execution against the reference database. Five
configurations, same questions, same harness — only the named component changes.

| metric | base | + retrieval | fine-tuned | FT + retrieval | **FT + repair** |
|---|---:|---:|---:|---:|---:|
| **strict execution accuracy** | 10.82 % | 9.27 % | 50.99 % | 41.72 % | **52.10 %** |
| projection-tolerant | 45.92 % | 43.93 % | 50.99 % | 41.72 % | **52.10 %** |
| executable SQL | 98.90 % | 92.27 % | 95.81 % | 94.92 % | **98.23 %** |
| schema hallucination | 0.66 % | 3.31 % | 1.99 % | 3.53 % | **0.66 %** |

Adapter: [`hari-krishna-ai/qwen3-8b-text2sql-qlora`](https://huggingface.co/hari-krishna-ai/qwen3-8b-text2sql-qlora)

### A property of this benchmark worth knowing before you use it

**35 % of the test set does not specify which columns to return.**
*"Show orders placed in 2023"* never says whether you want the order id alone or
five columns. So a model can retrieve exactly the right rows and still fail a
strict result-set comparison.

That is why two accuracy figures are reported. **Projection-tolerant accuracy**
wraps the prediction in a subquery and asks it for the gold columns, isolating
"wrong columns" from "wrong rows":

| failure mode | base model | fine-tuned |
|---|---:|---:|
| right rows, **wrong columns** | 159 | **0** |
| genuinely wrong rows | 240 | 203 |

Reporting strict accuracy alone would have made fine-tuning look like a
reasoning breakthrough. It is mostly the model learning this database's column
conventions — a real thing to learn, and not the same thing. **If you benchmark
on this dataset, report both numbers.**

---

## How it was built

1. **Templates** — parameterised question/SQL pairs across four domains and five
   difficulty tiers, each with multiple natural paraphrasings.
2. **Slot values sampled from the live database**, using interior quantiles so
   thresholds land inside the real data distribution. A benchmark full of
   predicates that match zero rows measures nothing.
3. **Every gold query executed** against the reference database; the result set
   is fingerprinted and stored. Queries that error are rejected, not shipped.
4. **Validation** — 3,045 of 3,045 examples pass parse, schema-grounding and
   execution checks.
5. **Split by template equivalence group.** All paraphrases and all slot
   variants of one template land in the same split, which is what makes the
   zero-template-overlap guarantee hold.

Deliberate inclusions: some questions correctly return **zero rows**, so a model
cannot assume every answer is non-empty.

---

## Limitations

Stated plainly, because they bound what a score on this dataset means.

- **Template-generated, not human-written.** Questions come from 74 / 52 / 48
  templates with paraphrasing. Real users write abbreviations, typos and
  genuinely ambiguous requests. **This is the largest caveat: a high score here
  does not establish robustness to real phrasing.**
- **One schema, 12 tables.** Small for an enterprise database. Schema-retrieval
  techniques that pay off at 200 tables *hurt* here — measured, twice.
- **Synthetic data.** Realistic distributions, but not real business data.
- **Single gold query per question.** Where several correct formulations exist,
  execution-fingerprint matching handles them; where a question is genuinely
  ambiguous about *columns*, use the projection-tolerant metric.
- **English only.**

## Intended use

Evaluating and fine-tuning text-to-SQL models where **executability and result
correctness** matter more than surface similarity. Also usable as SFT data — the
chat rendering used to train the adapter above is reproducible from these files
plus the prompt template, and is not shipped separately because it would
duplicate the 5.5 KB schema into every one of 2,133 records.

## Licence

Apache 2.0. All data synthetic; no personal information.
